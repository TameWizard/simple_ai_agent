import hashlib
import io
import logging
import os
import tempfile
import zipfile

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_test"
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

logging.basicConfig(level=logging.INFO)


token_splitter = SentenceTransformersTokenTextSplitter(
    model_name=EMBEDDING_MODEL,
    tokens_per_chunk=256,
    chunk_overlap=20,
)


def generate_sha256_hash() -> str:
    # Generate a random number
    random_data = os.urandom(16)
    # Create a SHA256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the random data
    sha256_hash.update(random_data)
    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def safe_get_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
):
    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' does not exist",
        )


def upsert_vector_data(
    collection: chromadb.api.models.Collection.Collection,
    zip_file: UploadFile,
) -> None:

    # Read ZIP into memory
    zip_file.file.seek(0)
    zip_bytes = io.BytesIO(zip_file.file.read())

    docs: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []

    with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
        with tempfile.TemporaryDirectory() as tmpdir:
            for info in zip_ref.infolist():
                filename = info.filename

                # Skip directories
                if info.is_dir():
                    continue

                _, ext = os.path.splitext(filename)
                if ext.lower() != ".pdf":
                    logging.warning(f"Skipping non-PDF file: {filename}")
                    continue

                logging.info(f"Processing PDF: {filename}")

                try:
                    extracted_path = zip_ref.extract(info, path=tmpdir)
                    loader = PyPDFLoader(extracted_path)
                    pages = loader.load_and_split(token_splitter)

                    for page in pages:
                        text = page.page_content.strip()
                        if not text:
                            continue

                        docs.append(text)
                        metas.append(page.metadata)
                        ids.append(generate_sha256_hash())

                except Exception as e:
                    logging.exception(f"Failed to process {filename}: {e}")

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No valid PDF content found in ZIP file",
        )

    # Batched upsert (fast + correct)
    collection.upsert(
        documents=docs,
        metadatas=metas,
        ids=ids,
    )

    logging.info(f"Upserted {len(docs)} document chunks into ChromaDB")

def reranker(query_text, docs, top_k=4):
    documents = docs["documents"][0]
    metadatas = docs["metadatas"][0]

    pairs = [[query_text, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)

    merged_list = list(zip(documents, metadatas, scores))
    sorted_list = sorted(merged_list, key=lambda x: x[2], reverse=True)
    top_results = sorted_list[:top_k]

    top_documents, document_names = zip(
        *(
            (doc, metadata.get("source", "").split("/")[-1])
            for doc, metadata, _ in top_results
        )
    )

    return top_documents, document_names
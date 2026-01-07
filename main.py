from typing import List, Annotated, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import duckdb
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Path, status, File, UploadFile, Form
from pydantic import BaseModel

from database_helpers import get_client, get_embedding_function, safe_get_collection, upsert_vector_data, reranker
from llm_helpers import text_to_sql, llm_generator

app = FastAPI()

class SQLUploadResult(BaseModel):
    table_name: str
    rows_inserted: int
    message: str

class ResponseModel(BaseModel):
    answer: str
    used_sql_query: str
    used_vector_sources: List[str]

@app.get("/")
async def root():
    return {"message": "Hello World"}


class CollectionInfo(BaseModel):
    name: Annotated[str, Path(description="Unique String ID of the collection on which it is will to be called")]
    documents_total: Annotated[int, Path(description="Current number of documents")]
    old_documents_total: Annotated[Optional[int] | None, Path(description="Number of documents "
                                                                          "before the current operation")] = None

@app.post("/upload_to_vector",
          status_code=status.HTTP_200_OK,
          response_model=CollectionInfo,
          name='Upload Vector Data',
          summary='Upload new data to the existing document vector collection',
          description="Upload new data from a zip file to the existing document collection",)
async def add_documents_to_collection(
        zip_file: Annotated[UploadFile, File(description='Path of the zip file to be added to the collection')],
        client: chromadb.PersistentClient = Depends(get_client),
        default_ef: embedding_functions.DefaultEmbeddingFunction = Depends(get_embedding_function)
) -> CollectionInfo:
    collection = client.get_or_create_collection(name='main_collection', embedding_function=default_ef)
    old_total = collection.count()
    upsert_vector_data(collection=collection, zip_file=zip_file)
    return CollectionInfo(name=collection.name, old_documents_total=old_total, documents_total=collection.count())


@app.post("/upload_to_sql/", response_model=SQLUploadResult)
async def upload_to_sql(
    csv_file: Annotated[UploadFile, File(description='CSV file to upload to SQL')]
) -> SQLUploadResult:
    table_name = "main_table"

    df = pd.read_csv(csv_file.file)

    with duckdb.connect("sql.db") as con:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("temp_df", df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")

        rows_inserted = con.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

    return SQLUploadResult(
        table_name=table_name,
        rows_inserted=rows_inserted,
        message=f"Successfully uploaded {rows_inserted} rows to {table_name}"
    )


@app.post("/query_the_agent",
          status_code=status.HTTP_200_OK,
          name="Query the Agent",
          summary="Ask AI the question and receive a response",
          description="Query the Agent with a message and receive a response",)
async def query_the_agent(query: Annotated[str,
        Form(description="Human Readable question to be sent to the Agent")],
            client: chromadb.PersistentClient = Depends(get_client),
            default_ef: embedding_functions.DefaultEmbeddingFunction = Depends(get_embedding_function)
                          ) -> ResponseModel:
    try:
        sql_response = await text_to_sql(query)
        collection = safe_get_collection(client=client, collection_name='main_collection')
        top_documents, document_names = reranker(
            query_text=query,
            docs=collection.query(
                query_texts=[query],
                n_results=10
            )
        )
        response = await llm_generator(query=query, sql_data=sql_response, vector_data=top_documents)
        return ResponseModel(answer=response, used_sql_query=sql_response.sql_query, used_vector_sources=document_names)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )

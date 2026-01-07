# Overview

A simple FastAPI AI agent with vector and SQL RAG implementation. The agent uses Gemini for LLM, DuckBD for SQL and 
ChromaDB for vector.

# Setup

The ***GEMINI_KEY*** environment variable must be set for the agent to work. <br> 
Optionally, you can provide ***LOGFIRE_TOKEN*** to monitor the LLM usage and responses. <br>
Run `uvicorn main:app --reload` to start the server.

# Usage

CSV file could be imported into DuckDB by using 
`curl -X POST http://localhost:8000/upload_to_sql/ \ -F "csv_file=@data.csv"` Note that for now, only a single CVS
file could exist in Agent's memory.<br>
The vector database could be created by running `curl -X POST http://localhost:8000/upload_to_vector \
  -F "zip_file=@documents.zip"` <br>
You can later query the agent by running `curl -X POST http://localhost:8000/query_the_agent \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=What is the price of Tesla, and how does it compare to the current macroeconomic trends?"`



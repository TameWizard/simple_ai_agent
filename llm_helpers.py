import os
from typing import Any

import duckdb
import logfire
import pandas as pd
import sqlglot
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from sqlglot.errors import ParseError

llm_key = str(os.environ.get('GEMINI_KEY'))
if not llm_key:
    raise EnvironmentError(
        "Missing GEMINI_KEY environment variable. "
        "Set it before starting the application."
    )

logfire_key = str(os.environ.get('LOGFIRE_TOKEN'))
if logfire_key:
    logfire.configure()
    logfire.instrument_pydantic_ai()

class DuckDBQuery(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def must_be_duckdb_compatible(cls, v):
        try:
            sqlglot.parse_one(v, read="duckdb")
        except ParseError as e:
            raise ValueError(f"Invalid DuckDB syntax: {e}")
        return v


class SQLResponseModel(BaseModel):
    sql_query: str
    sql_result: list[tuple[Any, ...]]


provider = GoogleProvider(api_key=llm_key)
main_model = GoogleModel('gemini-2.5-flash', provider=provider)
lite_model = GoogleModel('gemini-2.5-flash-lite', provider=provider)
fallback_model = FallbackModel(main_model, lite_model)


assistant_agent = Agent(lite_model, system_prompt="You are a Stock Investment Research Assistant. You will be given a "
                                            "collection of macroeconomic and strategic documents and "
                                            "SQL data containing stock information as well as user query and you will"
                                            "answer the query using the documents and SQL data.")

sql_agent = Agent(fallback_model, output_type=DuckDBQuery,
                                  system_prompt="You are a SQL query generator. A user query, SQL table description. "
                                                "Your task is to generate a single, accurate,"
                                                "syntactically correct DuckDB query that answers the natural language  "
                                                "request using the provided schema. Output ONLY the DuckDB SELECT , "
                                                "query with no comments, no explanations, and no additional text. "
                                                "The query must be read-only and must NOT contain any data modification "
                                                "operations such as INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE. "
                                                "The table is always called 'main_table'. Note that the company names "
                                                "are always with entity marks (like Inc, Ltd etc.) and no punctuation, "
                                                "like comma or periods")

def format_schema(schema):
    return "\n".join([f"{col[0]} ({col[1]})" for col in schema])

def format_sample_rows(rows, columns):
    df = pd.DataFrame(rows, columns=columns)
    return df.to_markdown(index=False)

async def text_to_sql(query: str) -> SQLResponseModel:
    with duckdb.connect("sql.db", read_only = True) as con:
        schema_raw = con.execute("DESCRIBE main_table;").fetchall()
        columns = [col[0] for col in schema_raw]
        schema_str = format_schema(schema_raw)

        sample_rows = con.execute("SELECT * FROM main_table LIMIT 3").fetchall()
        sample_rows_str = format_sample_rows(sample_rows, columns)
        llm_response = await sql_agent.run(f"User query: {query} "
                                           f"SQL table description: {sample_rows_str}")
        sql_query = llm_response.output.query
        print(sql_query)
        sql_result = con.execute(sql_query).fetchall()
    return SQLResponseModel(sql_query=sql_query, sql_result=sql_result)

async def llm_generator(query: str, sql_data: SQLResponseModel, vector_data: str) -> str:
    response = await assistant_agent.run(
        f'Macroeconomic and strategic documents: {vector_data} \n'
        f' SQL data: "Used SQL query:  {sql_data.sql_query} Retrieved SQL data: {sql_data.sql_result}" \n'
        f' User query: {query}'
    )
    return response.output

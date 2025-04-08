import chromadb
import pandas as pd
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import dotenv_values


config = dotenv_values(".env")
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


csv_file = f"{root_path}/llm/data/bbc-news-data-unique.csv"
df = pd.read_csv(csv_file, sep="\t", header=0)

titles = df.iloc[:, 1].tolist()
categories = df.iloc[:, 0].tolist()
print(titles)

# chromadb client
chroma_db_path = f"{root_path}/llm/chroma_dir"
client = chromadb.PersistentClient(path=chroma_db_path)

# collection
collection = client.get_collection(
    name="bbc-news-data",
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key=config["OPENAI_API_KEY"]
    )
)


def process_query_result(query_result):
    return [
        {query_result["documents"][0][nr]: query_result["metadatas"][0][nr]["category"]}
        for nr in range(len(query_result["ids"][0]))
    ]


if __name__ == "__main__":
    query_result = collection.query(query_texts=["China"], n_results=3)
    print(query_result)

    query_result = collection.query(query_texts=["Blog reading explodes in America"], n_results=3)
    print(query_result["documents"])

    query_result = collection.query(query_texts=["Uefa approves fake grass"], n_results=5)
    output = process_query_result(query_result)
    print(output)

    query_result = collection.query(query_texts=["economy recession"], n_results=5)
    output = process_query_result(query_result)
    print(output)

    query_result = collection.query(
        query_texts=["economy recession"],
        where={"category": "politics"},
        n_results=5
    )
    output = process_query_result(query_result)
    print(output)

    query_result = collection.query(
        query_texts=["economy recession"],
        where={"category": "business"},
        n_results=5
    )
    output = process_query_result(query_result)
    print(output)

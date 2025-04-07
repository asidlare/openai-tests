from dotenv import dotenv_values
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os
import pandas as pd


config = dotenv_values(".env")
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


csv_file = f"{root_path}/llm/data/bbc-news-data-unique.csv"
df = pd.read_csv(csv_file, sep="\t", header=0)

titles = df.iloc[:, 1].tolist()
ids = [f"doc{i}" for i in range(len(titles))]
categories = df.iloc[:, 0].tolist()
metadatas = [{'category': category} for category in categories]

# chromadb client
chroma_db_path = f"{root_path}/llm/chroma_dir"
client = chromadb.PersistentClient(path=chroma_db_path)

# client.delete_collection(name="bbc-news-data")
collection = client.get_or_create_collection(
    name="bbc-news-data",
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key=config["OPENAI_API_KEY"]
    )
)

# add collection
for it, i in enumerate(range(50, len(titles), 50)):
    collection.add(
        ids=ids[it * 50:i],
        documents=titles[it * 50:i],
        metadatas=metadatas[it * 50:i],
    )
import json
import os
import typing
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def create_embed_db(file_path: str, collection_name: str = 'crypto_docs') -> chromadb.Collection:
    # Load chunked data
    with open(file_path, "r") as f:
        text_chunks = json.load(f)

    # Initiate chromadb
    # Runtime version - Reembedding when restart
    # client = chromadb.Client(settings=Settings(anonymized_telemetry=False))

    # Save to local version
    client = PersistentClient(
        path="../db/chroma_store/",
        settings=Settings(anonymized_telemetry=False)
    )

    os.makedirs("../db/chroma_store", exist_ok=True)

    # Get or create db
    embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(collection_name, embedding_function=embedding_func)

    # Check is created then return
    if collection.count() > 0:
        print("Loaded existing Chroma collection.")
        return collection

    # Embedding all raw data
    ids = []
    documents = []
    metadatas = []

    count = 0
    for chunk in text_chunks:
        filename = chunk.get('filename', 'N/a')
        chunk_id = chunk.get('chunk_id', '0')
        text = chunk.get('text', 'N/a')

        ids.append(f'{filename}_{chunk_id}')
        documents.append(text)
        metadatas.append({
            "filename": filename,
            "chunk_id": chunk_id,
        })

        count += 1
        print(f'Process Embedding: {count} / {len(text_chunks)}')

    # Add to db by batch
    BATCH_SIZE = 5000
    total_ids = len(ids)

    for i in range(0, total_ids, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
        )
        print(f"Adding batch {i} to {i + len(batch_ids)} / {total_ids}")

    print("Chroma DB persisted to disk.")

    return collection


def retrieve_context(collection, query: str, top_k: int = 3) -> typing.List[str]:
    results = collection.query(
        query_texts=query,
        n_results=top_k,
    )

    return results

chroma = create_embed_db("../data/processed_chunks/chunks.json")
result = retrieve_context(chroma, "What is the role of 0x in decentralized trading?")
print(result)

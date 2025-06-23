import json
import typing
import chromadb
from sentence_transformers import SentenceTransformer


def create_embed_db(file_path: str):
    # Load chunked data
    with open(file_path, "r") as f:
        text_chunks = json.load(f)

    # Initiate embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initiate chromadb
    client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory="../db/chroma_store",  # or any folder you want
            anonymized_telemetry=False
        )
    )

    # Get or create db
    collection = client.get_or_create_collection("crypto_docs", embedding_function=None)

    # Check is created then return
    if collection.count() > 0:
        print("Loaded existing Chroma collection.")
        return collection

    # Embedding all raw data
    ids = []
    documents = []
    metadatas = []
    embeddings = []

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
        embeddings.append(model.encode(text))

        count += 1
        print(f'Process Embedding: {count} / {len(text_chunks)}')

    # Add to db by batch
    BATCH_SIZE = 5000
    total_ids = len(ids)

    for i in range(0, total_ids, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]
        batch_embeds = embeddings[i:i + BATCH_SIZE]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embeds,
        )
        print(f"Adding batch {i} to {i + len(batch_ids)} / {total_ids}")

    return collection


def search(collection, query: str, top_k: int = 3) -> typing.List[typing.Dict]:
    results = collection.query(
        query_texts=query,
        n_results=top_k,
    )

    return results


chroma = create_embed_db("../data/processed_chunks/chunks.json")
search(chroma, "What is the role of 0x in decentralized trading?")

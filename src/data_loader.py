import json
import os.path
import typing
import pdfplumber
from os import walk


def load_documents(folder_path) -> typing.List[typing.Dict]:
    docs = []

    # Get all filenames in folder_path
    files = next(walk(folder_path), (None, None, []))[2]  # [] if no file

    count = 0
    for file in files:

        count += 1
        print(f'Process: {count} / {len(files)}')

        if file.lower().endswith('.txt'):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                docs.append({"filename": file, "text": f.read()})
        elif file.endswith(".pdf"):
            try:
                with pdfplumber.open(str(os.path.join(folder_path, file))) as pdf:
                    full_text = ""

                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text
                    docs.append({"filename": file, "text": full_text})
            except Exception as e:
                print(f"[!] Error processing {file}: {e}")
                continue

    return docs


def chunk_text(text: str, chunk_size=500, overlap=100) -> typing.List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def save_chunks(docs: typing.List[typing.Dict]):
    result = []

    for doc in docs:
        text_chunks = chunk_text(doc['text'])

        for i, chunk in enumerate(text_chunks):
            result.append({
                "filename": doc.get('filename', 'N/A'),
                "chunk_id": i,
                "text": chunk,
            })

    with open('../data/processed_chunks/chunks.json', 'w+') as f:
        json.dump(result, f, indent=4)
    print(f"Sample chunk: {result[0]['text'][:200]}")


save_chunks(load_documents("../data/raw"))

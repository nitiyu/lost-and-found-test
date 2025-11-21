import streamlit as st
import openai

EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_embedding(text: str) -> list:
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in Streamlit secrets.")
    openai.api_key = key
    resp = openai.Embeddings.create(model=EMBEDDING_MODEL, input=text or "")
    return resp["data"][0]["embedding"]


def embedding_to_pgvector_literal(emb: list) -> str:
    # return string literal like '[0.1,0.2,...]'
    return "[" + ",".join(map(str, emb)) + "]"

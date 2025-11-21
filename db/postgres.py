import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st

def get_pg_conn():
    conn_str = st.secrets.get("PG_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("PG_CONNECTION_STRING not set in Streamlit secrets.")
    return psycopg2.connect(conn_str)


def init_db_postgres():
    """Create found_items table and enable vector extension."""
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS found_items (
                    id SERIAL PRIMARY KEY,
                    image_path TEXT,
                    subway_location TEXT,
                    color TEXT,
                    item_category TEXT,
                    item_type TEXT,
                    description TEXT,
                    embedding VECTOR(1536),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    contact_info TEXT
                );
                """
            )
        conn.commit()

# -----------------------------
# FILE: db/insert.py
# -----------------------------
from pathlib import Path
from db.postgres import get_pg_conn
from utils.embedding import get_openai_embedding, embedding_to_pgvector_literal
import streamlit as st


def add_found_item_postgres(data: dict, operator_contact: str = "", image_path: str = "") -> bool:
    description = data.get("description", "")
    try:
        emb = get_openai_embedding(description)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return False

    emb_literal = embedding_to_pgvector_literal(emb)

    sql = """
        INSERT INTO found_items (
            image_path, subway_location, color, item_category, item_type, description, embedding, contact_info
        ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
    """
    params = (
        image_path,
        ",".join(data.get("subway_location", [])),
        ",".join(data.get("color", [])),
        data.get("item_category", "null"),
        ",".join(data.get("item_type", [])),
        description,
        emb_literal,
        operator_contact,
    )

    try:
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error inserting into Postgres: {e}")
        return False
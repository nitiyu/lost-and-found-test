from db.postgres import get_pg_conn
from psycopg2.extras import RealDictCursor
from utils.embedding import get_openai_embedding, embedding_to_pgvector_literal
import streamlit as st


def search_found_items_postgres(user_report: dict, k: int = 5) -> list:
    sql = """
        SELECT id, image_path, subway_location, color, item_category, item_type, description,
               (embedding <-> %s::vector) AS distance
        FROM found_items
        WHERE 1=1
    """
    params = []

    icat = user_report.get("item_category")
    if icat and icat != "null":
        sql += " AND item_category = %s"
        params.append(icat)

    itypes = user_report.get("item_type", [])
    if itypes:
        sql += " AND item_type LIKE %s"
        params.append("%" + itypes[0] + "%")

    colors = user_report.get("color", [])
    if colors:
        sql += " AND color LIKE %s"
        params.append("%" + colors[0] + "%")

    slocs = user_report.get("subway_location", [])
    if slocs:
        sql += " AND subway_location LIKE %s"
        params.append("%" + slocs[0] + "%")

    try:
        user_emb = get_openai_embedding(user_report.get("description", ""))
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []

    user_emb_literal = embedding_to_pgvector_literal(user_emb)
    params = [user_emb_literal] + params

    sql += " ORDER BY distance ASC LIMIT %s"
    params.append(k)

    try:
        with get_pg_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        results = []
        for r in rows:
            dist = r.get("distance")
            similarity = None if dist is None else 1.0 / (1.0 + dist)
            results.append({
                "id": r["id"],
                "image_path": r["image_path"],
                "subway_location": r["subway_location"].split(",") if r["subway_location"] else [],
                "color": r["color"].split(",") if r["color"] else [],
                "item_category": r["item_category"],
                "item_type": r["item_type"].split(",") if r["item_type"] else [],
                "description": r["description"],
                "distance": dist,
                "similarity": similarity,
            })
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []
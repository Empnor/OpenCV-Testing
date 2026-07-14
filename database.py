"""
SQLite database for known people, their reference photos, and precomputed
face embeddings. Photos themselves stay on disk (known_faces/<name>/...);
this database stores the metadata and the embedding vectors so they don't
need to be recomputed every time the app starts.
"""

import sqlite3
import numpy as np
from collections import defaultdict

DB_PATH = "faces.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            file_path TEXT NOT NULL,
            embedding BLOB,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def add_person(name):
    """Insert a person if they don't already exist. Returns their id either way."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO people (name) VALUES (?)", (name,))
    conn.commit()
    cur.execute("SELECT id FROM people WHERE name = ?", (name,))
    person_id = cur.fetchone()[0]
    conn.close()
    return person_id


def add_photo(person_id, file_path, embedding):
    """Store a photo's path and its precomputed embedding (or NULL if no face found)."""
    conn = get_connection()
    embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    conn.execute(
        "INSERT INTO photos (person_id, file_path, embedding) VALUES (?, ?, ?)",
        (person_id, file_path, embedding_blob)
    )
    conn.commit()
    conn.close()


def get_all_people_with_embeddings():
    """Return {name: averaged_embedding} for use in live recognition."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT people.name, photos.embedding
        FROM photos
        JOIN people ON people.id = photos.person_id
        WHERE photos.embedding IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    embeddings_by_name = defaultdict(list)
    for name, embedding_blob in rows:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)
        embeddings_by_name[name].append(embedding)

    result = {}
    for name, embeddings_list in embeddings_by_name.items():
        result[name] = np.mean(np.array(embeddings_list), axis=0)
    return result


def list_people():
    """Return [(id, name, photo_count), ...] for display in the admin panel."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT people.id, people.name, COUNT(photos.id) as photo_count
        FROM people
        LEFT JOIN photos ON photos.person_id = people.id
        GROUP BY people.id
        ORDER BY people.name
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def list_photos_for_person(person_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, file_path FROM photos WHERE person_id = ? ORDER BY date_added",
        (person_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def delete_person(person_id):
    """Deletes the person and all their photo records (not the files on disk)."""
    conn = get_connection()
    conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()


def delete_photo(photo_id):
    conn = get_connection()
    conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    conn.commit()
    conn.close()
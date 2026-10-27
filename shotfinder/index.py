"""SQLite index: schema, CRUD and queries. The only module that touches SQLite."""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

import sqlite_vec
import xxhash

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  hash TEXT NOT NULL,
  kind TEXT NOT NULL,
  size INTEGER, mtime REAL, created REAL,
  status TEXT NOT NULL,
  error TEXT,
  stages_done TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);

-- text from OCR and speech; ts = NULL for images, seconds for video
CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(
  content, file_id UNINDEXED, ts UNINDEXED, source UNINDEXED,
  tokenize = 'unicode61 remove_diacritics 2'
);

-- one vector per image, one per kept video frame
CREATE TABLE IF NOT EXISTS vec_meta (
  rowid INTEGER PRIMARY KEY,
  file_id INTEGER,
  ts REAL
);
CREATE INDEX IF NOT EXISTS idx_vec_meta_file_id ON vec_meta(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[512]);
"""

_HASH_CHUNK = 1024 * 1024
_VECTOR_DIMENSIONS = 512
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def compute_file_hash(path: str | Path) -> str:
    """Return xxhash of the first 1 MB plus the file size (survives move/rename)."""
    path = Path(path)
    size = path.stat().st_size
    digest = xxhash.xxh3_64()
    with path.open("rb") as handle:
        digest.update(handle.read(_HASH_CHUNK))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def _normalise_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else path.absolute()


def _fts_query(query: str) -> str:
    """Turn free text into an FTS5 query where every token is a prefix."""
    tokens = _TOKEN_RE.findall(query)
    return " AND ".join(f'"{token}"*' for token in tokens)


def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Index:
    """Thread-safe facade over one SQLite file, with one connection per thread."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        try:
            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @property
    def conn(self) -> sqlite3.Connection:
        connection = getattr(self._local, "conn", None)
        if connection is None:
            connection = self._connect()
            self._local.conn = connection
        return connection

    def _ensure_schema(self) -> None:
        with self._schema_lock:
            if self._schema_ready:
                return
            self.conn.executescript(SCHEMA)
            self._schema_ready = True

    def close(self) -> None:
        """Close the connection owned by the current thread."""
        connection = getattr(self._local, "conn", None)
        if connection is not None:
            connection.close()
            self._local.conn = None

    # -- files ---------------------------------------------------------------

    def upsert_file(self, path: str | Path, kind: str) -> tuple[int, bool]:
        """Insert or refresh a file.

        Returns ``(file_id, needs_indexing)``. A file whose hash matches a row
        whose old path no longer exists is treated as moved: its path is updated
        and ``needs_indexing`` is False. A copy (old path still exists) gets its
        own row.
        """
        path = _normalise_path(path)
        stat = path.stat()
        key = str(path)

        with self.conn as connection:
            existing = connection.execute(
                "SELECT * FROM files WHERE path = ?", (key,)
            ).fetchone()
            if existing is not None:
                # Fast path: an unchanged file never needs its 1 MB re-read.
                if (
                    existing["size"] == stat.st_size
                    and existing["mtime"] == stat.st_mtime
                ):
                    if (
                        existing["created"] != stat.st_ctime
                        or existing["kind"] != kind
                    ):
                        connection.execute(
                            "UPDATE files SET created = ?, kind = ? WHERE id = ?",
                            (stat.st_ctime, kind, existing["id"]),
                        )
                    return int(existing["id"]), False

                digest = compute_file_hash(path)
                if existing["hash"] == digest and existing["size"] == stat.st_size:
                    connection.execute(
                        "UPDATE files SET mtime = ?, created = ?, kind = ? WHERE id = ?",
                        (stat.st_mtime, stat.st_ctime, kind, existing["id"]),
                    )
                    return int(existing["id"]), False

                # Same path, different content: invalidate every derived row.
                self._delete_content(connection, int(existing["id"]))
                connection.execute(
                    """
                    UPDATE files
                       SET hash = ?, kind = ?, size = ?, mtime = ?, created = ?,
                           status = 'pending', error = NULL, stages_done = ''
                     WHERE id = ?
                    """,
                    (
                        digest,
                        kind,
                        stat.st_size,
                        stat.st_mtime,
                        stat.st_ctime,
                        existing["id"],
                    ),
                )
                return int(existing["id"]), True

            digest = compute_file_hash(path)
            same_hash = connection.execute(
                "SELECT * FROM files WHERE hash = ? ORDER BY id", (digest,)
            ).fetchall()
            moved = next(
                (row for row in same_hash if not Path(row["path"]).exists()), None
            )
            if moved is not None:
                connection.execute(
                    "UPDATE files SET path = ?, kind = ?, size = ?, mtime = ?, created = ? WHERE id = ?",
                    (
                        key,
                        kind,
                        stat.st_size,
                        stat.st_mtime,
                        stat.st_ctime,
                        moved["id"],
                    ),
                )
                return int(moved["id"]), False

            cursor = connection.execute(
                """
                INSERT INTO files
                    (path, hash, kind, size, mtime, created, status, error, stages_done)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, '')
                """,
                (
                    key,
                    digest,
                    kind,
                    stat.st_size,
                    stat.st_mtime,
                    stat.st_ctime,
                ),
            )
            return int(cursor.lastrowid), True

    def find_by_hash(self, digest: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM files WHERE hash = ? ORDER BY id LIMIT 1", (digest,)
        ).fetchone()
        return _as_dict(row)

    def find_by_path(self, path: str | Path) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM files WHERE path = ?", (str(_normalise_path(path)),)
        ).fetchone()
        return _as_dict(row)

    def get_file(self, file_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return _as_dict(row)

    def mark_status(
        self,
        file_id: int,
        status: str,
        error: str | None = None,
        stages_done: str | None = None,
        expected_hash: str | None = None,
    ) -> bool:
        """Update a file's status, optionally only if its content hash still matches."""
        with self.conn as connection:
            if stages_done is None:
                sql = "UPDATE files SET status = ?, error = ? WHERE id = ?"
                parameters: tuple[Any, ...] = (status, error, file_id)
            else:
                sql = "UPDATE files SET status = ?, error = ?, stages_done = ? WHERE id = ?"
                parameters = (status, error, stages_done, file_id)
            if expected_hash is not None:
                sql += " AND hash = ?"
                parameters += (expected_hash,)
            cursor = connection.execute(sql, parameters)
            return cursor.rowcount > 0

    def delete_file(self, file_id: int) -> None:
        """Delete a file row and all of its text and vector rows."""
        with self.conn as connection:
            self._delete_content(connection, file_id)
            connection.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def _delete_content(self, connection: sqlite3.Connection, file_id: int) -> None:
        connection.execute("DELETE FROM text_fts WHERE file_id = ?", (file_id,))
        self._delete_vectors(connection, file_id)

    def _delete_vectors(
        self, connection: sqlite3.Connection, file_id: int
    ) -> None:
        rows = connection.execute(
            "SELECT rowid FROM vec_meta WHERE file_id = ?", (file_id,)
        ).fetchall()
        for row in rows:
            connection.execute("DELETE FROM vec_items WHERE rowid = ?", (row["rowid"],))
        connection.execute("DELETE FROM vec_meta WHERE file_id = ?", (file_id,))

    # -- queue ---------------------------------------------------------------

    def next_pending(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM files
             WHERE status = 'pending'
             ORDER BY CASE kind WHEN 'image' THEN 0 WHEN 'video' THEN 1 ELSE 2 END, id
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def count_pending(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE status = 'pending'"
        ).fetchone()
        return int(row[0])

    def count_files(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()
        return int(row[0])

    def next_vision_backfill(self, limit: int = 100) -> list[dict[str, Any]]:
        """Images that finished OCR before the vision models were available."""
        rows = self.conn.execute(
            """
            SELECT * FROM files
             WHERE status = 'done' AND kind = 'image'
               AND ',' || COALESCE(stages_done, '') || ',' NOT LIKE '%,vision,%'
             ORDER BY id
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # -- text ----------------------------------------------------------------

    def add_text_rows(
        self,
        file_id: int,
        rows: Iterable[str | tuple[str, float | None]],
        *,
        source: str,
        expected_hash: str | None = None,
    ) -> int:
        """Replace the rows for one file/source with new text.

        Strings become rows with ``ts = NULL``; ``(text, ts)`` pairs are used for
        timed speech segments. Replacing makes a retried stage idempotent. When
        ``expected_hash`` is given, nothing is written if the file changed while
        the extractor was running.
        """
        normalised: list[tuple[str, int, float | None, str]] = []
        for row in rows:
            if isinstance(row, str):
                text, timestamp = row, None
            else:
                text, timestamp = row
            text = text.strip()
            if text:
                normalised.append((text, file_id, timestamp, source))
        if not normalised:
            return 0
        with self.conn as connection:
            if expected_hash is not None:
                unchanged = connection.execute(
                    "SELECT 1 FROM files WHERE id = ? AND hash = ?",
                    (file_id, expected_hash),
                ).fetchone()
                if unchanged is None:
                    return 0
            connection.execute(
                "DELETE FROM text_fts WHERE file_id = ? AND source = ?",
                (file_id, source),
            )
            connection.executemany(
                "INSERT INTO text_fts (content, file_id, ts, source) VALUES (?, ?, ?, ?)",
                normalised,
            )
        return len(normalised)

    def search_text(self, query: str, limit: int = 40) -> list[dict[str, Any]]:
        """Prefix-match every query token, ordered by BM25 (best first)."""
        match = _fts_query(query)
        if not match:
            return []
        rows = self.conn.execute(
            """
            SELECT f.id AS file_id, f.path, f.kind,
                   text_fts.ts AS ts, text_fts.source AS source,
                   snippet(text_fts, 0, '[', ']', '…', 12) AS snippet,
                   bm25(text_fts) AS score
              FROM text_fts
              JOIN files f ON f.id = text_fts.file_id
             WHERE text_fts MATCH ?
             ORDER BY score, f.id
             LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_files(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id AS file_id, path, kind,
                   NULL AS ts, NULL AS source, NULL AS snippet, 0.0 AS score
              FROM files
             WHERE status = 'done'
             ORDER BY COALESCE(mtime, 0) DESC, id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # -- vectors -------------------------------------------------------------

    def add_vectors(
        self,
        file_id: int,
        items: Iterable[tuple[float | None, Sequence[float]]],
        *,
        expected_hash: str | None = None,
    ) -> int:
        """Replace one file's vectors; returns how many were written.

        Vectors must be L2-normalized 512-float embeddings. Replacing makes a
        retried stage idempotent, and ``expected_hash`` rejects results computed
        from a file that changed in the meantime.
        """
        prepared: list[tuple[float | None, list[float]]] = []
        for timestamp, vector in items:
            values = [float(value) for value in vector]
            if len(values) != _VECTOR_DIMENSIONS:
                raise ValueError(
                    f"expected {_VECTOR_DIMENSIONS}-dimensional vector, got {len(values)}"
                )
            prepared.append((timestamp, values))
        if not prepared:
            return 0
        with self.conn as connection:
            if expected_hash is not None:
                unchanged = connection.execute(
                    "SELECT 1 FROM files WHERE id = ? AND hash = ?",
                    (file_id, expected_hash),
                ).fetchone()
                if unchanged is None:
                    return 0
            self._delete_vectors(connection, file_id)
            for timestamp, values in prepared:
                cursor = connection.execute(
                    "INSERT INTO vec_meta (file_id, ts) VALUES (?, ?)",
                    (file_id, timestamp),
                )
                connection.execute(
                    "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(values)),
                )
        return len(prepared)

    def search_vectors(
        self, vector: Sequence[float], k: int = 50
    ) -> list[dict[str, Any]]:
        """Return the *k* nearest rows, ascending by L2 distance.

        ``vec0`` uses L2 by default and stored vectors are L2-normalized, so
        cosine similarity is ``1 - distance ** 2 / 2``.
        """
        if k <= 0:
            return []
        values = [float(value) for value in vector]
        if len(values) != _VECTOR_DIMENSIONS:
            raise ValueError(
                f"expected {_VECTOR_DIMENSIONS}-dimensional vector, got {len(values)}"
            )
        rows = self.conn.execute(
            """
            SELECT knn.rowid AS rowid, m.file_id AS file_id, m.ts AS ts,
                   f.path AS path, f.kind AS kind, knn.distance AS distance
              FROM (SELECT rowid, distance FROM vec_items
                     WHERE embedding MATCH ? AND k = ?) AS knn
              JOIN vec_meta m ON m.rowid = knn.rowid
              JOIN files f ON f.id = m.file_id
             ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(values), k),
        ).fetchall()
        return [dict(row) for row in rows]

"""
Grep 风格全文检索 — SQLite FTS5
类似 Claude 的 glob+grep，对精确参数、型号等关键词效果好。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence


class GrepIndex:
    """
    SQLite FTS5 全文索引，按 col_name（如 nio_ec6）分车型存储 chunk 文本。
    """

    TABLE = "chunks_fts"
    SCHEMA_VERSION = 2  # v1 was contentless (columns unreadable)

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            root = Path(__file__).resolve().parents[2]
            db_path = str(root / "grep_index.db")
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and busy timeout."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self):
        conn = self._connect()
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS _grep_schema (version INTEGER)")
            cur = conn.execute("SELECT version FROM _grep_schema LIMIT 1")
            row = cur.fetchone()
            current = row[0] if row else 0
            if current < self.SCHEMA_VERSION:
                conn.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
                conn.execute("DELETE FROM _grep_schema")
                conn.execute("INSERT INTO _grep_schema (version) VALUES (?)", (self.SCHEMA_VERSION,))
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.TABLE}
                USING fts5(
                    col_name UNINDEXED,
                    chunk_text,
                    tokenize='unicode61'
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def has_chunks(self, col_name: str) -> bool:
        """检查该 col 是否已有数据"""
        conn = self._connect()
        try:
            cur = conn.execute(
                f"SELECT 1 FROM {self.TABLE} WHERE col_name = ? LIMIT 1",
                (col_name,),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    def insert_chunks(self, col_name: str, chunks: Sequence[str]) -> int:
        """
        插入该车型的 chunks。会先删除该 col 的旧数据再插入。
        """
        conn = self._connect()
        try:
            conn.execute(
                f"DELETE FROM {self.TABLE} WHERE col_name = ?",
                (col_name,),
            )
            for t in chunks:
                conn.execute(
                    f"INSERT INTO {self.TABLE}(col_name, chunk_text) VALUES(?, ?)",
                    (col_name, t),
                )
            conn.commit()
            return len(chunks)
        finally:
            conn.close()

    def search(
        self,
        col_name: str,
        keywords: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        按关键词搜索。keywords 支持空格分隔多个词，FTS5 默认 OR 逻辑。
        返回 [{"text": ..., "rank": ...}, ...]
        """
        # FTS5 保留字需转义；多个词用 OR 连接
        terms = []
        for w in keywords.strip().split():
            w = w.strip()
            if not w:
                continue
            # 简单转义：若含 " 或 - 等，用双引号包裹
            if any(c in w for c in '"*-'):
                terms.append(f'"{w}"')
            else:
                terms.append(w)
        if not terms:
            return []

        match_expr = " OR ".join(terms)

        conn = self._connect()
        try:
            # FTS5: col_name 过滤 + MATCH
            cur = conn.execute(
                f"""
                SELECT chunk_text, bm25({self.TABLE}) as rank
                FROM {self.TABLE}
                WHERE col_name = ? AND {self.TABLE} MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (col_name, match_expr, limit),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError as e:
            if "syntax error" in str(e).lower() or "malformed" in str(e).lower():
                return []
            raise
        finally:
            conn.close()

        return [{"text": r[0], "rank": round(r[1], 4)} for r in rows]


def get_grep_index(db_path: str | None = None) -> GrepIndex:
    """获取全局 GrepIndex 实例"""
    global _grep_index
    if _grep_index is None:
        _grep_index = GrepIndex(db_path)
    return _grep_index


_grep_index: GrepIndex | None = None

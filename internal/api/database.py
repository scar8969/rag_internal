import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid
import hashlib


@dataclass
class User:
    id: int
    username: str
    email: Optional[str]
    password_hash: str
    created_at: datetime
    is_admin: bool
    is_active: bool
    daily_limit: int

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "is_admin": self.is_admin,
            "is_active": self.is_active,
            "daily_limit": self.daily_limit,
        }


@dataclass
class UsageLog:
    id: int
    user_id: int
    endpoint: str
    tokens_used: int
    timestamp: datetime


class Database:
    def __init__(self, db_path: str = "./data/app.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_admin BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                daily_limit INTEGER DEFAULT 1000
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                key_hash TEXT NOT NULL,
                name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME,
                expires_at DATETIME,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                endpoint TEXT NOT NULL,
                method TEXT,
                status_code INTEGER,
                tokens_used INTEGER DEFAULT 0,
                latency_ms INTEGER,
                ip_address TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                source_path TEXT,
                file_size INTEGER,
                chunk_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_logs_user ON usage_logs(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_logs_timestamp ON usage_logs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id)
        """)

        conn.commit()
        conn.close()

    def create_user(
        self,
        username: str,
        password_hash: str,
        email: Optional[str] = None,
        is_admin: bool = False,
    ) -> User:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, is_admin)
            VALUES (?, ?, ?, ?)
            """,
            (username, email, password_hash, is_admin),
        )
        conn.commit()

        user_id = cursor.lastrowid
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_user(row)

    def get_user_by_username(self, username: str) -> Optional[User]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_user(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_user(row) if row else None

    def list_users(self) -> list[User]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_user(row) for row in rows]

    def update_user(self, user_id: int, **kwargs) -> bool:
        allowed = ["email", "is_admin", "is_active", "daily_limit", "password_hash"]
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return False

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        cursor.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?",
            list(updates.values()) + [user_id],
        )

        conn = self._get_connection()
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    def delete_user(self, user_id: int) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    def log_usage(
        self,
        user_id: int,
        endpoint: str,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        tokens_used: int = 0,
        latency_ms: Optional[int] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO usage_logs (user_id, endpoint, method, status_code, tokens_used, latency_ms, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, endpoint, method, status_code, tokens_used, latency_ms, ip_address),
        )
        conn.commit()
        conn.close()

    def get_usage_for_user(
        self,
        user_id: int,
        days: int = 7,
    ) -> list[UsageLog]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM usage_logs
            WHERE user_id = ? AND timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
            """,
            (user_id, days),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_usage(row) for row in rows]

    def get_usage_stats(self, user_id: int) -> dict:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as total_requests,
                   COALESCE(SUM(tokens_used), 0) as total_tokens,
                   COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM usage_logs
            WHERE user_id = ? AND timestamp >= datetime('now', '-7 days')
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()

        return {
            "total_requests": row["total_requests"],
            "total_tokens": row["total_tokens"],
            "active_days": row["active_days"],
        }

    def create_api_key(self, user_id: int, name: str) -> str:
        key_id = str(uuid.uuid4())
        key = f"rag_{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO api_keys (id, user_id, key_hash, name)
            VALUES (?, ?, ?, ?)
            """,
            (key_id, user_id, key_hash, name),
        )
        conn.commit()
        conn.close()

        return key

    def verify_api_key(self, key: str) -> Optional[User]:
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.* FROM users u
            JOIN api_keys ak ON u.id = ak.user_id
            WHERE ak.key_hash = ? AND ak.is_active = TRUE AND u.is_active = TRUE
            """,
            (key_hash,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            cursor.execute(
                "UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE key_hash = ?",
                (key_hash,),
            )

        return self._row_to_user(row) if row else None

    def get_user_api_keys(self, user_id: int) -> list[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, created_at, last_used, is_active FROM api_keys WHERE user_id = ?",
            (user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def revoke_api_key(self, key_id: str, user_id: int) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE api_keys SET is_active = FALSE WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    def _row_to_user(self, row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            is_admin=bool(row["is_admin"]),
            is_active=bool(row["is_active"]),
            daily_limit=row["daily_limit"],
        )

    def _row_to_usage(self, row: sqlite3.Row) -> UsageLog:
        return UsageLog(
            id=row["id"],
            user_id=row["user_id"],
            endpoint=row["endpoint"],
            tokens_used=row["tokens_used"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if isinstance(row["timestamp"], str) else row["timestamp"],
        )

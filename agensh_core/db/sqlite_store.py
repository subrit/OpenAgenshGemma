"""
High-performance asynchronous SQLite storage engine for Agensh.
Replaces MySQL, Gitea DB, and Mattermost DB with an embedded, zero-maintenance,
WAL-mode SQLite database supporting 4 to 1024 concurrent agents.
Uses persistent connection caching to achieve blazing-fast query execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

logger = logging.getLogger("agensh.sqlite")


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 60000;
PRAGMA temp_store = MEMORY;

-- Board gists (append-only shared context)
CREATE TABLE IF NOT EXISTS board_gists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    ts_epoch REAL NOT NULL,
    expires_at REAL,
    delegation_id TEXT,
    detail TEXT,
    verified_checkpoint INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_board_ts ON board_gists(board_id, ts_epoch);
CREATE INDEX IF NOT EXISTS ix_board_type ON board_gists(board_id, type);
CREATE INDEX IF NOT EXISTS ix_board_delegation ON board_gists(delegation_id);

-- Read cursors per agent
CREATE TABLE IF NOT EXISTS board_cursor (
    agent TEXT NOT NULL,
    board_id TEXT NOT NULL,
    gist_id INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (agent, board_id)
);

-- Shown notices / acknowledgments
CREATE TABLE IF NOT EXISTS shown (
    event_id INTEGER NOT NULL,
    agent TEXT NOT NULL,
    shown_at REAL NOT NULL,
    PRIMARY KEY (event_id, agent)
);
CREATE INDEX IF NOT EXISTS ix_shown_agent ON shown(agent);

-- Pending leases for mid-turn delivery
CREATE TABLE IF NOT EXISTS pending_leases (
    agent TEXT PRIMARY KEY,
    lease_token TEXT NOT NULL,
    lease_until REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_lease_until ON pending_leases(lease_until);

-- Messaging (replaces Mattermost: channels and direct messages)
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL DEFAULT 'general',
    from_agent TEXT NOT NULL,
    to_agent TEXT,
    content TEXT NOT NULL,
    msg_type TEXT NOT NULL DEFAULT 'chat',
    created_at REAL NOT NULL,
    read_at REAL
);
CREATE INDEX IF NOT EXISTS ix_msg_channel ON messages(channel, created_at);
CREATE INDEX IF NOT EXISTS ix_msg_target ON messages(to_agent, read_at);

-- Key-value store for agent session persistence
CREATE TABLE IF NOT EXISTS agent_kv (
    agent TEXT NOT NULL,
    k TEXT NOT NULL,
    v TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (agent, k)
);

-- Task claims & coordination
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    claimed_by TEXT,
    claimed_at REAL,
    expires_at REAL,
    result TEXT
);
CREATE INDEX IF NOT EXISTS ix_task_status ON tasks(status);

-- Router event queue
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key TEXT NOT NULL,
    source TEXT NOT NULL,
    agent TEXT NOT NULL,
    payload TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    sent_at REAL,
    last_error TEXT,
    batch_id TEXT,
    next_retry_at REAL,
    UNIQUE (dedup_key, agent)
);
CREATE INDEX IF NOT EXISTS ix_events_agent_state ON events(agent, state, id);
CREATE INDEX IF NOT EXISTS ix_events_state_retry ON events(state, next_retry_at);

-- Sandbox execution logs
CREATE TABLE IF NOT EXISTS sandbox_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    command TEXT NOT NULL,
    exit_code INTEGER,
    stdout TEXT,
    stderr TEXT,
    duration_ms REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sandbox_agent ON sandbox_logs(agent, created_at);
"""


class SQLiteStore:
    """Async thread-safe SQLite store with WAL mode for Agensh."""

    def __init__(self, db_path: Path | str = "agensh.db"):
        self.db_path = str(Path(db_path).resolve())
        self._write_lock = asyncio.Lock()
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize tables and indexes with WAL mode and cache open connection."""
        if self._db is not None:
            return

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # Explicitly configure WAL mode and memory cache
        await self._db.execute("PRAGMA journal_mode = WAL;")
        await self._db.execute("PRAGMA synchronous = NORMAL;")
        await self._db.execute("PRAGMA busy_timeout = 60000;")
        await self._db.execute("PRAGMA temp_store = MEMORY;")
        await self._db.execute("PRAGMA cache_size = -64000;")

        async with self._write_lock:
            await self._db.executescript(SCHEMA_SQL)
            await self._db.commit()

        logger.info(f"SQLiteStore initialized at {self.db_path}")

    async def get_connection(self) -> aiosqlite.Connection:
        """Return the active connection."""
        if self._db is None:
            await self.initialize()
        return self._db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # -------------------------------------------------------------------------
    # Board Gists Operations
    # -------------------------------------------------------------------------

    async def insert_gist(
        self,
        board_id: str,
        thread_id: str,
        gist_type: str,
        content: str,
        ts_epoch: float,
        expires_at: Optional[float] = None,
        delegation_id: Optional[str] = None,
        detail: Optional[str] = None,
        verified_checkpoint: bool = False,
    ) -> int:
        """Insert a new admitted gist into the board log."""
        await self.initialize()
        sql = """
            INSERT INTO board_gists (
                board_id, thread_id, type, content, ts_epoch,
                expires_at, delegation_id, detail, verified_checkpoint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            cursor = await self._db.execute(
                sql,
                (
                    board_id,
                    thread_id,
                    gist_type,
                    content,
                    ts_epoch,
                    expires_at,
                    delegation_id,
                    detail,
                    1 if verified_checkpoint else 0,
                ),
            )
            await self._db.commit()
            return int(cursor.lastrowid)

    async def get_recent_gists(
        self, board_id: str, limit: int = 2000
    ) -> List[Dict[str, Any]]:
        """Retrieve recent gists for a board."""
        await self.initialize()
        sql = """
            SELECT id, board_id, thread_id, type, content, ts_epoch,
                   expires_at, delegation_id, detail, verified_checkpoint
            FROM board_gists
            WHERE board_id = ?
            ORDER BY id ASC
        """
        cursor = await self._db.execute(sql, (board_id,))
        rows = await cursor.fetchall()

        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "label": f"G{r['id']}",
                    "board_id": r["board_id"],
                    "thread_id": r["thread_id"],
                    "type": r["type"],
                    "content": r["content"],
                    "ts": float(r["ts_epoch"]),
                    "expires_at": float(r["expires_at"]) if r["expires_at"] is not None else None,
                    "delegation_id": r["delegation_id"],
                    "has_detail": bool(r["detail"]),
                    "verified_checkpoint": bool(r["verified_checkpoint"]),
                }
            )
        return out[-limit:] if len(out) > limit else out

    async def get_all_gists_raw(self) -> List[Dict[str, Any]]:
        """Fetch all gists to hydrate in-memory state on startup."""
        await self.initialize()
        sql = """
            SELECT id, board_id, thread_id, type, content, ts_epoch,
                   expires_at, delegation_id, detail, verified_checkpoint
            FROM board_gists
            ORDER BY id ASC
        """
        cursor = await self._db.execute(sql)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_detail(self, board_id: str, gist_id: int) -> Optional[str]:
        """Fetch full detail for a G<id> note."""
        await self.initialize()
        sql = "SELECT detail FROM board_gists WHERE board_id = ? AND id = ?"
        cursor = await self._db.execute(sql, (board_id, gist_id))
        row = await cursor.fetchone()
        return row[0] if row and row[0] else None

    async def grep_gists(
        self,
        board_id: str,
        where_clause: str,
        params: List[Any],
        limit: int = 500,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Execute parameterized grep search over board history."""
        await self.initialize()
        count_sql = f"SELECT COUNT(*) FROM board_gists WHERE board_id = ? AND {where_clause}"
        data_sql = f"""
            SELECT id, board_id, thread_id, type, content, ts_epoch, delegation_id
            FROM board_gists
            WHERE board_id = ? AND {where_clause}
            ORDER BY id DESC
            LIMIT ?
        """
        full_params = [board_id] + params
        cursor = await self._db.execute(count_sql, full_params)
        count_row = await cursor.fetchone()
        total_matches = count_row[0] if count_row else 0

        cursor = await self._db.execute(data_sql, full_params + [limit])
        rows = await cursor.fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "label": f"G{r['id']}",
                    "id": r["id"],
                    "thread_id": r["thread_id"],
                    "type": r["type"],
                    "content": r["content"],
                    "ts": float(r["ts_epoch"]),
                    "delegation_id": r["delegation_id"],
                }
            )
        return results, total_matches

    # -------------------------------------------------------------------------
    # Cursor and Snapshot Operations
    # -------------------------------------------------------------------------

    async def get_board_cursor(self, agent: str, board_id: str) -> int:
        await self.initialize()
        sql = "SELECT gist_id FROM board_cursor WHERE agent = ? AND board_id = ?"
        cursor = await self._db.execute(sql, (agent, board_id))
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def advance_board_cursor(
        self, agent: str, board_id: str, gist_id: int
    ) -> None:
        if gist_id <= 0:
            return
        await self.initialize()
        sql = """
            INSERT INTO board_cursor (agent, board_id, gist_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent, board_id) DO UPDATE SET
                gist_id = MAX(board_cursor.gist_id, excluded.gist_id),
                updated_at = excluded.updated_at
        """
        async with self._write_lock:
            await self._db.execute(sql, (agent, board_id, gist_id, time.time()))
            await self._db.commit()

    async def get_snapshot_entries(
        self, agent: str, board_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch unread gists for an agent across boards, advancing cursors."""
        await self.initialize()
        result: Dict[str, List[Dict[str, Any]]] = {}

        for bid in board_ids:
            cur_id = await self.get_board_cursor(agent, bid)
            sql = """
                SELECT id, board_id, thread_id, type, content, ts_epoch,
                       expires_at, delegation_id, detail, verified_checkpoint
                FROM board_gists
                WHERE board_id = ? AND id > ?
                ORDER BY id ASC
            """
            cursor = await self._db.execute(sql, (bid, cur_id))
            rows = await cursor.fetchall()

            entries = []
            max_id = cur_id
            for r in rows:
                gid = int(r["id"])
                max_id = max(max_id, gid)
                entries.append(
                    {
                        "id": gid,
                        "label": f"G{gid}",
                        "thread_id": r["thread_id"],
                        "type": r["type"],
                        "content": r["content"],
                        "ts": float(r["ts_epoch"]),
                        "expires_at": float(r["expires_at"]) if r["expires_at"] is not None else None,
                        "delegation_id": r["delegation_id"],
                        "has_detail": bool(r["detail"]),
                    }
                )

            if max_id > cur_id:
                await self.advance_board_cursor(agent, bid, max_id)

            result[bid] = entries

        return result

    # -------------------------------------------------------------------------
    # Mid-turn Leases & Notice Acknowledgment
    # -------------------------------------------------------------------------

    async def get_pending_notices(
        self, agent: str, board_ids: List[str], lease_seconds: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Lease unread peer gists and unread direct messages to agent."""
        await self.initialize()
        notices: List[Dict[str, Any]] = []

        # 1. Unread direct messages
        msg_sql = """
            SELECT id, from_agent, to_agent, content, created_at, msg_type
            FROM messages
            WHERE to_agent = ? AND read_at IS NULL
            ORDER BY id ASC
        """
        cursor = await self._db.execute(msg_sql, (agent,))
        msg_rows = await cursor.fetchall()

        for mr in msg_rows:
            notices.append(
                {
                    "event_id": mr["id"],
                    "kind": "direct_message",
                    "from_agent": mr["from_agent"],
                    "content": mr["content"],
                    "msg_type": mr["msg_type"],
                    "ts": mr["created_at"],
                }
            )

        # 2. Unread peer gists
        for bid in board_ids:
            cur_id = await self.get_board_cursor(agent, bid)
            gist_sql = """
                SELECT g.id, g.thread_id, g.type, g.content, g.ts_epoch, g.delegation_id
                FROM board_gists g
                LEFT JOIN shown s ON s.event_id = g.id AND s.agent = ?
                WHERE g.board_id = ? AND g.id > ? AND g.thread_id != ? AND s.event_id IS NULL
                ORDER BY g.id ASC
                LIMIT 16
            """
            cursor = await self._db.execute(gist_sql, (agent, bid, cur_id, agent))
            g_rows = await cursor.fetchall()

            for gr in g_rows:
                notices.append(
                    {
                        "event_id": gr["id"],
                        "kind": "peer_gist",
                        "label": f"G{gr['id']}",
                        "from_agent": gr["thread_id"],
                        "type": gr["type"],
                        "content": gr["content"],
                        "delegation_id": gr["delegation_id"],
                        "ts": gr["ts_epoch"],
                    }
                )

        return notices

    async def mark_shown(self, agent: str, event_ids: List[int]) -> None:
        """Mark notice events as acknowledged."""
        if not event_ids:
            return
        await self.initialize()
        now = time.time()
        sql = "INSERT OR IGNORE INTO shown (event_id, agent, shown_at) VALUES (?, ?, ?)"
        msg_ack_sql = "UPDATE messages SET read_at = ? WHERE id = ? AND to_agent = ?"

        async with self._write_lock:
            for eid in event_ids:
                await self._db.execute(sql, (eid, agent, now))
                await self._db.execute(msg_ack_sql, (now, eid, agent))
            await self._db.commit()

    # -------------------------------------------------------------------------
    # Messaging Operations (Mattermost Replacement)
    # -------------------------------------------------------------------------

    async def send_message(
        self,
        from_agent: str,
        to_agent: Optional[str],
        channel: str,
        content: str,
        msg_type: str = "chat",
    ) -> int:
        """Send a channel message or direct message."""
        await self.initialize()
        sql = """
            INSERT INTO messages (channel, from_agent, to_agent, content, msg_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            cursor = await self._db.execute(
                sql,
                (channel, from_agent, to_agent, content, msg_type, time.time()),
            )
            await self._db.commit()
            return int(cursor.lastrowid)

    async def get_messages(
        self,
        agent: Optional[str] = None,
        channel: Optional[str] = None,
        limit: int = 50,
        since_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent channel or direct messages."""
        await self.initialize()
        clauses = ["id > ?"]
        params: List[Any] = [since_id]

        if channel:
            clauses.append("channel = ?")
            params.append(channel)
        if agent:
            clauses.append("(to_agent = ? OR from_agent = ? OR to_agent IS NULL)")
            params.extend([agent, agent])

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT id, channel, from_agent, to_agent, content, msg_type, created_at, read_at
            FROM messages
            WHERE {where_sql}
            ORDER BY id ASC
            LIMIT ?
        """
        params.append(limit)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Task Coordination (Gitea Replacement)
    # -------------------------------------------------------------------------

    async def create_task(
        self, task_id: str, title: str, description: str = ""
    ) -> None:
        await self.initialize()
        sql = """
            INSERT INTO tasks (task_id, title, description, status)
            VALUES (?, ?, ?, 'open')
            ON CONFLICT(task_id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description
        """
        async with self._write_lock:
            await self._db.execute(sql, (task_id, title, description))
            await self._db.commit()

    async def claim_task(
        self, task_id: str, agent: str, ttl_seconds: float = 3600.0
    ) -> Tuple[bool, str]:
        """Claim a task atomically with TTL."""
        await self.initialize()
        now = time.time()
        expires = now + ttl_seconds

        async with self._write_lock:
            cursor = await self._db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            task = await cursor.fetchone()

            if not task:
                await self._db.execute(
                    "INSERT INTO tasks (task_id, title, status, claimed_by, claimed_at, expires_at) "
                    "VALUES (?, ?, 'claimed', ?, ?, ?)",
                    (task_id, task_id, agent, now, expires),
                )
                await self._db.commit()
                return True, "Task created and claimed."

            if (
                task["status"] == "open"
                or (task["expires_at"] and task["expires_at"] < now)
                or task["claimed_by"] == agent
            ):
                await self._db.execute(
                    "UPDATE tasks SET status = 'claimed', claimed_by = ?, claimed_at = ?, expires_at = ? "
                    "WHERE task_id = ?",
                    (agent, now, expires, task_id),
                )
                await self._db.commit()
                return True, f"Claimed task '{task_id}' until {expires:.0f}."
            else:
                return False, f"Task '{task_id}' already claimed by {task['claimed_by']}."

    async def list_tasks(self) -> List[Dict[str, Any]]:
        await self.initialize()
        cursor = await self._db.execute("SELECT * FROM tasks ORDER BY task_id")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Sandbox Execution Logs
    # -------------------------------------------------------------------------

    async def log_sandbox(
        self,
        agent: str,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_ms: float,
    ) -> None:
        await self.initialize()
        sql = """
            INSERT INTO sandbox_logs (agent, command, exit_code, stdout, stderr, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            await self._db.execute(
                sql,
                (agent, command, exit_code, stdout, stderr, duration_ms, time.time()),
            )
            await self._db.commit()

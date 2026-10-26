"""
Unified Board and Coordination Service for Agensh.
Coordinates SQLite storage and in-memory DeLM-style shared lessons.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..db.sqlite_store import SQLiteStore
from .lessons import (
    DEFAULT_CLAIM_TTL_SECONDS,
    MAX_CONTENT_CHARS,
    MAX_PATCH_SUMMARY_CHARS,
    VALID_TYPES,
    SharedLessons,
    SharedLessonsError,
)
from .query import compile_sqlite_sql, parse as parse_query

logger = logging.getLogger("agensh.board")


class BoardService:
    """Provides high-performance shared context, claims, and messaging."""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self.boards: Dict[str, SharedLessons] = {}
        self._lock = asyncio.Lock()
        self._hydrated = False

    def get_board(self, board_id: str) -> SharedLessons:
        if board_id not in self.boards:
            self.boards[board_id] = SharedLessons(task_id=board_id)
        return self.boards[board_id]

    async def hydrate(self) -> None:
        """Replay SQLite history into memory on startup."""
        if self._hydrated:
            return
        await self.store.initialize()
        rows = await self.store.get_all_gists_raw()
        for r in rows:
            b = self.get_board(r["board_id"])
            gid = int(r["id"])
            entry = {
                "id": gid,
                "_label": f"G{gid}",
                "ts": float(r["ts_epoch"]),
                "thread_id": r["thread_id"],
                "type": r["type"],
                "content": r["content"],
                "delegation_id": r.get("delegation_id"),
                "_expires_at": float(r["expires_at"]) if r["expires_at"] is not None else None,
                "_has_detail": bool(r.get("detail")),
                "detail": r.get("detail"),
                "verified_checkpoint": bool(r.get("verified_checkpoint")),
            }
            b.admit_entry(entry)
        self._hydrated = True
        logger.info(f"BoardService hydrated {len(rows)} entries from SQLite")

    async def post_note(
        self,
        board_id: str,
        thread_id: str,
        note_type: str,
        content: str,
        delegation_id: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        detail: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post a note to the shared board."""
        await self.hydrate()
        note_type = note_type.strip().upper()
        if note_type not in VALID_TYPES:
            raise SharedLessonsError(f"invalid type {note_type!r}; must be one of {VALID_TYPES}")

        content = (content or "").strip()
        if not content:
            raise SharedLessonsError("content is empty after strip()")

        type_cap = MAX_PATCH_SUMMARY_CHARS if note_type == "PATCH_SUMMARY" else MAX_CONTENT_CHARS
        content = content[:type_cap]

        now_ts = time.time()
        expires_at = None
        if note_type == "CLAIM":
            ttl = float(ttl_seconds) if ttl_seconds is not None else DEFAULT_CLAIM_TTL_SECONDS
            expires_at = now_ts + max(0.0, ttl)

        # 1. Persist in SQLite
        b = self.get_board(board_id)
        verified_cp = b._is_verified_patch_summary_checkpoint(content) if note_type == "PATCH_SUMMARY" else False

        gid = await self.store.insert_gist(
            board_id=board_id,
            thread_id=thread_id,
            gist_type=note_type,
            content=content,
            ts_epoch=now_ts,
            expires_at=expires_at,
            delegation_id=delegation_id,
            detail=detail,
            verified_checkpoint=verified_cp,
        )

        # 2. Admit into memory
        entry = {
            "id": gid,
            "_label": f"G{gid}",
            "ts": now_ts,
            "thread_id": thread_id,
            "type": note_type,
            "content": content,
            "delegation_id": delegation_id,
            "_expires_at": expires_at,
            "_has_detail": bool(detail),
            "detail": detail,
            "_verified_checkpoint": verified_cp,
        }
        b.admit_entry(entry)

        return {
            "status": "ok",
            "gist_id": gid,
            "label": f"G{gid}",
            "type": note_type,
            "expires_at": expires_at,
        }

    async def read_board(self, board_id: str, raw: bool = False) -> List[Dict[str, Any]]:
        """Read live view or raw retained history."""
        await self.hydrate()
        b = self.get_board(board_id)
        if raw:
            return list(b._entries)
        return b.get_live_entries()

    async def grep_board(
        self,
        board_id: str,
        query_expr: str,
        types: Optional[List[str]] = None,
        limit: int = 500,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Search board using boolean expression & types."""
        await self.hydrate()
        where_parts = []
        params: List[Any] = []

        if query_expr and query_expr.strip():
            ast = parse_query(query_expr.strip())
            sql_frag, p = compile_sqlite_sql(ast)
            where_parts.append(sql_frag)
            params.extend(p)
        else:
            where_parts.append("1=1")

        if types:
            placeholders = ",".join("?" for _ in types)
            where_parts.append(f"type IN ({placeholders})")
            params.extend(types)

        where_clause = " AND ".join(where_parts)
        return await self.store.grep_gists(board_id, where_clause, params, limit)

    async def unfold(self, board_id: str, gist_id: int) -> Optional[str]:
        """Fetch long detail behind a G-label."""
        await self.hydrate()
        return await self.store.get_detail(board_id, gist_id)

    async def get_snapshot(
        self, agent: str, board_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch unread gists for agent and advance cursors."""
        await self.hydrate()
        return await self.store.get_snapshot_entries(agent, board_ids)

    async def get_pending_notices(
        self, agent: str, board_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch pending notices (direct messages and peer gists) for mid-turn delivery."""
        await self.hydrate()
        return await self.store.get_pending_notices(agent, board_ids)

    async def acknowledge_notices(self, agent: str, event_ids: List[int]) -> None:
        """Acknowledge notices delivered to the agent."""
        await self.store.mark_shown(agent, event_ids)

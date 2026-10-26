"""
Maintains the recent shared-context view: typed notes, expiring claims,
patch status, conservative deduplication and prompt rendering.
Adapted from DeLM and Agensh's shared lessons protocol.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

VALID_TYPES = ("FACT", "OBSERVED", "FAIL", "CLAIM", "PATCH_SUMMARY")
MAX_CONTENT_CHARS = 100
MAX_PATCH_SUMMARY_CHARS = 300
MAX_BOARD_ENTRIES = 2000
DEFAULT_CLAIM_TTL_SECONDS = 3600.0

_NONE_TOKENS = ("(none)", "none", "-", "—")
_PATCH_SUMMARY_WEAK_EVIDENCE = (
    "pending", "not verified", "unverified", "should", "will verify",
    "to be verified", "tbd", "looks", "seems",
)
_PATCH_SUMMARY_SOLID_EVIDENCE_RE = re.compile(
    r"\b(pass(?:ed|es)?|verified|confirmed|no longer|now returns|now produces)\b",
    re.IGNORECASE,
)


def parse_patch_summary_field(content: str, field_name: str) -> Optional[str]:
    prefix = field_name.lower() + "="
    for part in str(content or "").split("|"):
        s = part.strip()
        if s.lower().startswith(prefix):
            return s[len(prefix):].strip()
    return None


class SharedLessonsError(ValueError):
    pass


class SharedLessons:
    """In-memory recent view with TTL pruning and patch status tracking."""

    def __init__(self, task_id: str, max_entries: int = MAX_BOARD_ENTRIES):
        self.task_id = task_id
        self.max_entries = max_entries
        self._entries: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._start_time = time.time()

    def admit_entry(self, entry: Dict[str, Any]) -> None:
        """Admit an entry replayed from SQLite or freshly created."""
        type_ = entry.get("type", "")
        content = entry.get("content", "")

        if type_ == "PATCH_SUMMARY" and self._is_verified_patch_summary_checkpoint(content):
            entry["_verified_checkpoint"] = True

        # Replace author's earlier unverified patch summary with same delegation_id
        author = entry.get("thread_id")
        delegation = entry.get("delegation_id")
        if type_ == "PATCH_SUMMARY" and delegation and author:
            self._entries = [
                e for e in self._entries
                if not (
                    e.get("type") == "PATCH_SUMMARY"
                    and e.get("thread_id") == author
                    and e.get("delegation_id") == delegation
                    and not e.get("_verified_checkpoint")
                )
            ]

        # If a FAIL is published by an author for delegation_id, mark prior unverified patch as rejected
        if type_ == "FAIL" and delegation and author:
            for e in self._entries:
                if (
                    e.get("type") == "PATCH_SUMMARY"
                    and e.get("thread_id") == author
                    and e.get("delegation_id") == delegation
                    and not e.get("_verified_checkpoint")
                ):
                    e["_invalidated"] = True

        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def _is_verified_patch_summary_checkpoint(self, content: str) -> bool:
        ev = parse_patch_summary_field(content, "evidence")
        if not ev:
            return False
        ev_norm = ev.strip().lower()
        for weak in _PATCH_SUMMARY_WEAK_EVIDENCE:
            if weak in ev_norm:
                return False
        return bool(_PATCH_SUMMARY_SOLID_EVIDENCE_RE.search(ev_norm))

    def get_live_entries(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get entries that are active (non-expired claims, non-invalidated patches)."""
        t = now if now is not None else time.time()
        live = []
        for e in self._entries:
            if e.get("_invalidated"):
                continue
            if e.get("type") == "CLAIM":
                exp = e.get("_expires_at") or e.get("expires_at")
                if exp and exp < t:
                    continue
            live.append(e)
        return live

    def render_prompt_context(self, now: Optional[float] = None, max_lines: int = 50) -> str:
        """Render recent board state into prompt section."""
        entries = self.get_live_entries(now)
        if not entries:
            return "==== SHARED CONTEXT ====\n(No board entries yet)\n========================"

        lines = ["==== SHARED CONTEXT ===="]
        for e in entries[-max_lines:]:
            label = e.get("_label") or f"G{e.get('id', '')}"
            type_ = e.get("type", "")
            content = e.get("content", "")
            author = e.get("thread_id", "")
            detail_flag = " (+detail)" if e.get("_has_detail") or e.get("has_detail") else ""
            lines.append(f"{label} [{type_}] ({author}) {content}{detail_flag}")
        lines.append("========================")
        return "\n".join(lines)

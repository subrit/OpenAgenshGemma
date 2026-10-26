"""
Agensh (Open Source Edition): Scalable Multi-Agent Harness (4 to 1024 Agents).
Powered by Google Gemma 4, E2B Sandboxes, SQLite, and Node.js.
"""

from .config import AgenshConfig
from .db.sqlite_store import SQLiteStore
from .board.service import BoardService
from .llm.gemma4_client import Gemma4Client
from .sandbox.e2b_manager import E2BSandboxManager
from .agent.worker import AgentWorker, AgentState
from .orchestrator.engine import ScalableHarness

__version__ = "2.0.0"

__all__ = [
    "AgenshConfig",
    "SQLiteStore",
    "BoardService",
    "Gemma4Client",
    "E2BSandboxManager",
    "AgentWorker",
    "AgentState",
    "ScalableHarness",
]

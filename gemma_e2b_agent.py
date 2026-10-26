#!/usr/bin/env python3
"""
Standalone Gemma 4 + E2B Sandbox Agent Worker for Agensh.
Connects to the SQLite shared context board and executes tasks in an E2B sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from agensh_core.agent.worker import AgentWorker
from agensh_core.board.service import BoardService
from agensh_core.db.sqlite_store import SQLiteStore
from agensh_core.llm.gemma4_client import Gemma4Client
from agensh_core.sandbox.e2b_manager import E2BSandboxManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gemma_agent")


def parse_args():
    parser = argparse.ArgumentParser(description="Agensh Gemma 4 + E2B Agent")
    parser.add_argument("--agent", default=os.environ.get("AGENT_NAME", "worker-1"), help="Agent name")
    parser.add_argument("--total-workers", type=int, default=int(os.environ.get("PBENCH_WORKERS", 8)), help="Team size (4-1024)")
    parser.add_argument("--db", default=os.environ.get("SQLITE_DB_PATH", "agensh.db"), help="SQLite DB path")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gemma4"), help="Model name")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "ollama"), help="LLM Provider")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:11434"), help="LLM URL")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "dummy"), help="LLM API Key")
    parser.add_argument("--mock-llm", action="store_true", default=os.environ.get("GEMMA_MOCK", "0") in ("1", "true"))
    parser.add_argument("--e2b-api-key", default=os.environ.get("E2B_API_KEY", None))
    parser.add_argument("--use-local-sandbox", action="store_true", default=False)
    parser.add_argument("--turns", type=int, default=10, help="Number of turns to execute")
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID", "pbench-pandoc"))
    return parser.parse_args()


async def main():
    args = parse_args()
    logger.info(f"Initializing {args.agent} (Team size: {args.total_workers}, Model: {args.model})...")

    store = SQLiteStore(args.db)
    await store.initialize()

    board = BoardService(store)
    await board.hydrate()

    llm = Gemma4Client(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        mock_mode=args.mock_llm,
    )

    sandbox = E2BSandboxManager(
        api_key=args.e2b_api_key,
        use_local=args.use_local_sandbox,
    )

    worker = AgentWorker(
        agent_id=args.agent,
        total_workers=args.total_workers,
        task_id=args.task_id,
        task_brief="Analyze reference behavior, claim subtasks, verify with tests, and share findings.",
        board_service=board,
        llm_client=llm,
        sandbox_manager=sandbox,
        store=store,
    )

    for turn in range(1, args.turns + 1):
        logger.info(f"[{args.agent}] Starting turn {turn}/{args.turns}...")
        res = await worker.run_turn()
        logger.info(f"[{args.agent}] Turn {turn} completed: {res.get('thought', '')[:80]}...")
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())

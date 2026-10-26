"""
Comprehensive Test Suite for Agensh Open Source Architecture.
Verifies:
- SQLite Storage Layer (WAL mode, async queries, claims, messages)
- Board Service (DeLM-style shared context, TTL pruning, grep, unfold)
- Gemma 4 LLM Client (prompts, tool generation, simulated fallback)
- E2B Sandbox Manager (isolated commands, file read/write, code execution)
- Multi-Agent Scaling (scaling 4, 16, 64, up to 1024 agents)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

# Add root paths
repo_root = Path(__file__).resolve().parents[1]
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agensh_core.config import AgenshConfig
from agensh_core.db.sqlite_store import SQLiteStore
from agensh_core.board.service import BoardService
from agensh_core.board.lessons import SharedLessons
from agensh_core.board.query import parse_query, compile_sqlite_sql
from agensh_core.llm.gemma4_client import Gemma4Client
from agensh_core.sandbox.e2b_manager import E2BSandboxManager, LocalSandbox
from agensh_core.agent.worker import AgentWorker
from agensh_core.orchestrator.engine import ScalableHarness


class TestAgenshSQLite(unittest.IsolatedAsyncioTestCase):
    """Test SQLite database operations and concurrency."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_agensh.db"
        self.store = SQLiteStore(self.db_path)
        await self.store.initialize()

    async def asyncTearDown(self):
        await self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_wal_mode_and_tables(self):
        conn = await self.store.get_connection()
        async with conn.execute("PRAGMA journal_mode;") as cursor:
            row = await cursor.fetchone()
            self.assertEqual(row[0].lower(), "wal")
        await conn.close()

    async def test_gist_insert_and_retrieval(self):
        gid = await self.store.insert_gist(
            board_id="test-board",
            thread_id="worker-1",
            gist_type="FACT",
            content="Gemma 4 successfully booted in sandbox",
            ts_epoch=time.time(),
            delegation_id="comp_boot",
            detail="Full booting traceback and log verified.",
        )
        self.assertGreater(gid, 0)

        recent = await self.store.get_recent_gists("test-board")
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["type"], "FACT")
        self.assertEqual(recent[0]["label"], f"G{gid}")

        detail = await self.store.get_detail("test-board", gid)
        self.assertIn("traceback and log verified", detail)

    async def test_concurrent_writes_100(self):
        """Verify high concurrency without database lock errors."""
        async def _write(idx: int):
            return await self.store.insert_gist(
                board_id="test-board",
                thread_id=f"worker-{idx % 16}",
                gist_type="OBSERVED" if idx % 2 == 0 else "FAIL",
                content=f"Observation {idx}",
                ts_epoch=time.time(),
            )

        tasks = [_write(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        self.assertEqual(len(results), 100)
        self.assertTrue(all(r > 0 for r in results))

    async def test_messaging_and_dms(self):
        mid = await self.store.send_message(
            from_agent="worker-1",
            to_agent="worker-2",
            channel="direct",
            content="Collision: I am already claiming parser.",
            msg_type="collision",
        )
        self.assertGreater(mid, 0)

        notices = await self.store.get_pending_notices("worker-2", ["test-board"])
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0]["from_agent"], "worker-1")
        self.assertIn("Collision", notices[0]["content"])

        await self.store.mark_shown("worker-2", [mid])
        notices_after = await self.store.get_pending_notices("worker-2", ["test-board"])
        self.assertEqual(len(notices_after), 0)


class TestBoardService(unittest.IsolatedAsyncioTestCase):
    """Test DeLM board rules, TTL pruning, and grep queries."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_board.db"
        self.store = SQLiteStore(self.db_path)
        self.board = BoardService(self.store)
        await self.board.hydrate()

    async def asyncTearDown(self):
        await self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_claim_and_ttl_expiration(self):
        # Post claim with 1s TTL
        res = await self.board.post_note(
            board_id="pbench-top",
            thread_id="worker-1",
            note_type="CLAIM",
            content="CLAIM: parser module",
            delegation_id="parser",
            ttl_seconds=1.0,
        )
        self.assertEqual(res["status"], "ok")

        live = await self.board.read_board("pbench-top", raw=False)
        self.assertEqual(len(live), 1)

        # Wait for TTL to expire
        await asyncio.sleep(1.2)
        live_expired = await self.board.read_board("pbench-top", raw=False)
        self.assertEqual(len(live_expired), 0)

        # Raw retained history still preserves the entry
        raw = await self.board.read_board("pbench-top", raw=True)
        self.assertEqual(len(raw), 1)

    async def test_boolean_grep(self):
        await self.board.post_note("pbench-top", "worker-1", "FACT", "parser syntax ast nodes")
        await self.board.post_note("pbench-top", "worker-2", "FAIL", "codegen ast failed on call")
        await self.board.post_note("pbench-top", "worker-3", "PATCH_SUMMARY", "files=eval.py | idea=fix | evidence=passed confirmed")

        # Search AND: ast & parser
        res, count = await self.board.grep_board("pbench-top", "ast & parser")
        self.assertEqual(count, 1)
        self.assertIn("syntax", res[0]["content"])

        # Search OR: codegen | eval
        res_or, count_or = await self.board.grep_board("pbench-top", "codegen | eval")
        self.assertEqual(count_or, 2)


class TestE2BSandbox(unittest.IsolatedAsyncioTestCase):
    """Test E2B Sandbox functionality and Local Sandbox isolation."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sandbox_manager = E2BSandboxManager(
            api_key=None,
            base_dir=self.temp_dir,
            use_local=True,
        )

    async def asyncTearDown(self):
        await self.sandbox_manager.cleanup_all()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_sandbox_file_and_exec(self):
        sandbox = await self.sandbox_manager.get_sandbox("worker-test")

        # Write file
        await sandbox.write_file("main.py", "print('Hello from E2B Sandbox')\n")
        content = await sandbox.read_file("main.py")
        self.assertEqual(content.strip(), "print('Hello from E2B Sandbox')")

        # Exec command
        code, stdout, stderr = await sandbox.exec_cmd("python main.py")
        self.assertEqual(code, 0)
        self.assertIn("Hello from E2B Sandbox", stdout)

        # Run code directly
        stdout2, stderr2, res2 = await sandbox.run_code("print(1337 * 2)")
        self.assertIn("2674", stdout2)


class TestGemma4Client(unittest.IsolatedAsyncioTestCase):
    """Test Gemma 4 client, tool call parsing, and simulated mode."""

    async def test_simulated_gemma_turn(self):
        client = Gemma4Client(mock_mode=True)
        messages = [
            {"role": "system", "content": "You are worker-1"},
            {"role": "user", "content": "Begin turn 1"},
        ]
        result = await client.generate_turn(
            agent_name="worker-1",
            messages=messages,
            turn_number=1,
            context_hints={"unclaimed_tasks": ["parser"]},
        )
        self.assertIn("tool_calls", result)
        self.assertGreater(len(result["tool_calls"]), 0)
        first_tool = result["tool_calls"][0]
        self.assertEqual(first_tool["name"], "board_write")
        self.assertEqual(first_tool["arguments"]["type"], "CLAIM")


class TestMultiAgentScaling(unittest.IsolatedAsyncioTestCase):
    """Test scaling multi-agent harness across 4, 16, 64, and up to 1024 agents."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "scale_test.db"

    async def asyncTearDown(self):
        if hasattr(self, "store") and self.store:
            await self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_scale_4_agents(self):
        config = AgenshConfig(
            workers=4,
            max_active_workers=4,
            sqlite_db_path=self.db_path,
            mock_llm=True,
            use_local_sandbox=True,
            sandbox_base_dir=Path(self.temp_dir) / "sandboxes_4",
        )
        harness = ScalableHarness(config)
        try:
            summary = await harness.run_rounds(rounds=2)
            self.assertEqual(summary["workers_count"], 4)
            self.assertEqual(summary["total_turns"], 8)
            self.assertGreaterEqual(summary["board_total_entries"], 4)
        finally:
            await harness.store.close()

    async def test_scale_32_agents(self):
        config = AgenshConfig(
            workers=32,
            max_active_workers=16,
            sqlite_db_path=self.db_path,
            mock_llm=True,
            use_local_sandbox=True,
            sandbox_base_dir=Path(self.temp_dir) / "sandboxes_32",
        )
        harness = ScalableHarness(config)
        try:
            summary = await harness.run_rounds(rounds=2)
            self.assertEqual(summary["workers_count"], 32)
            self.assertEqual(summary["total_turns"], 64)
            self.assertGreaterEqual(summary["board_total_entries"], 32)
        finally:
            await harness.store.close()

    async def test_scale_1024_agents_simulation(self):
        """Stress-test harness managing 1024 agents simultaneously with concurrency budgeting."""
        config = AgenshConfig(
            workers=1024,
            max_active_workers=128, # 128 active concurrency slots
            sqlite_db_path=self.db_path,
            mock_llm=True,
            use_local_sandbox=True,
            sandbox_base_dir=Path(self.temp_dir) / "sandboxes_1024",
        )
        harness = ScalableHarness(config)
        await harness.initialize()

        self.assertEqual(len(harness.workers), 1024)
        self.assertEqual(harness.workers_count, 1024)
        self.assertEqual(harness.max_active, 128)

        # Run 1 complete round across all 1024 agents
        start = time.time()
        try:
            summary = await harness.run_rounds(rounds=1)
            duration = time.time() - start

            self.assertEqual(summary["workers_count"], 1024)
            self.assertEqual(summary["total_turns"], 1024)
            self.assertGreaterEqual(summary["board_total_entries"], 1024)
            print(f"\n[BENCHMARK] 1,024 agents executed in {duration:.2f}s (~{1024/duration:.1f} turns/sec)")
        finally:
            await harness.store.close()


if __name__ == "__main__":
    unittest.main()

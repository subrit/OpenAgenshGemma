"""
Scalable Multi-Agent Orchestrator Harness for Agensh.
Supports scalable fleets from 4 to 1024 autonomous agents with concurrency budgeting,
SQLite WAL persistence, Gemma 4 reasoning, and E2B sandbox execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..agent.worker import AgentWorker, AgentState
from ..board.service import BoardService
from ..config import AgenshConfig
from ..db.sqlite_store import SQLiteStore
from ..llm.gemma4_client import Gemma4Client
from ..sandbox.e2b_manager import E2BSandboxManager

logger = logging.getLogger("agensh.orchestrator")


class ScalableHarness:
    """Manages execution, scheduling, and metrics for 4 to 1024 collaborating agents."""

    def __init__(self, config: AgenshConfig):
        self.config = config
        self.workers_count = max(4, min(1024, config.workers))
        self.max_active = min(self.workers_count, config.max_active_workers)

        # Core components
        self.store = SQLiteStore(config.sqlite_db_path)
        self.board_service = BoardService(self.store)
        self.llm_client = Gemma4Client(
            provider=config.llm_provider,
            model=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            max_concurrency=config.llm_max_concurrency,
            rpm_limit=config.llm_rpm_limit,
            mock_mode=config.mock_llm,
        )
        self.sandbox_manager = E2BSandboxManager(
            api_key=config.e2b_api_key,
            base_dir=config.sandbox_base_dir,
            use_local=config.use_local_sandbox,
            pool_sandboxes=(self.workers_count > 64),
            max_sandboxes=min(32, self.max_active),
        )

        # Agent workers
        self.workers: List[AgentWorker] = []
        self._semaphore = asyncio.Semaphore(self.max_active)
        self._running = False
        self._stop_event = asyncio.Event()

        # Metrics
        self.start_time = 0.0
        self.total_turns_completed = 0
        self.total_patches_verified = 0

    async def initialize(self) -> None:
        """Initialize database, board service, and instantiate worker fleet."""
        await self.store.initialize()
        await self.board_service.hydrate()

        self.workers.clear()
        for idx in range(1, self.workers_count + 1):
            agent_id = f"worker-{idx}"
            worker = AgentWorker(
                agent_id=agent_id,
                total_workers=self.workers_count,
                task_id=self.config.task_id,
                task_brief=self.config.task_brief,
                board_service=self.board_service,
                llm_client=self.llm_client,
                sandbox_manager=self.sandbox_manager,
                store=self.store,
            )
            self.workers.append(worker)

        logger.info(
            f"ScalableHarness initialized: {self.workers_count} agents configured "
            f"(concurrency limit: {self.max_active} active coroutines)."
        )

    async def run_rounds(self, rounds: int = 5, on_turn_done: Optional[Callable] = None) -> Dict[str, Any]:
        """Run a coordinated sequence of turns across all 4-1024 agents."""
        await self.initialize()
        self.start_time = time.time()
        self._running = True

        logger.info(f"Starting {rounds} rounds across {self.workers_count} agents...")

        for round_idx in range(1, rounds + 1):
            if self._stop_event.is_set():
                break

            round_start = time.time()
            logger.info(f"--- Round {round_idx}/{rounds} starting ({self.workers_count} agents) ---")

            # Execute turn for each worker with concurrency semaphore
            tasks = [self._execute_worker_turn(w, on_turn_done) for w in self.workers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            round_duration = time.time() - round_start
            logger.info(
                f"--- Round {round_idx}/{rounds} completed in {round_duration:.2f}s "
                f"(Total turns so far: {self.total_turns_completed}) ---"
            )

        self._running = False
        return await self.get_summary()

    async def _execute_worker_turn(
        self, worker: AgentWorker, on_turn_done: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute a single agent's turn throttled by the active concurrency semaphore."""
        async with self._semaphore:
            res = await worker.run_turn()
            self.total_turns_completed += 1
            if on_turn_done:
                try:
                    on_turn_done(res)
                except Exception:
                    pass
            return res

    async def run_continuous(self, duration_seconds: float = 60.0) -> Dict[str, Any]:
        """Run continuous asynchronous loops across all agents until duration expires."""
        await self.initialize()
        self.start_time = time.time()
        self._running = True
        self._stop_event.clear()

        logger.info(
            f"Running continuous harness for {duration_seconds}s with {self.workers_count} agents..."
        )

        async def _worker_loop(w: AgentWorker):
            while not self._stop_event.is_set():
                async with self._semaphore:
                    await w.run_turn()
                    self.total_turns_completed += 1
                await asyncio.sleep(0.05)

        tasks = [asyncio.create_task(_worker_loop(w)) for w in self.workers]

        try:
            await asyncio.wait_for(
                asyncio.shield(self._stop_event.wait()), timeout=duration_seconds
            )
        except asyncio.TimeoutError:
            pass
        finally:
            self._stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._running = False

        return await self.get_summary()

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False

    async def get_summary(self) -> Dict[str, Any]:
        """Compute aggregated statistics across the multi-agent run."""
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        all_gists = await self.board_service.read_board("pbench-top", raw=True)
        live_gists = await self.board_service.read_board("pbench-top", raw=False)

        claims = [g for g in live_gists if g.get("type") == "CLAIM"]
        patches = [g for g in all_gists if g.get("type") == "PATCH_SUMMARY"]
        fails = [g for g in all_gists if g.get("type") == "FAIL"]
        facts = [g for g in all_gists if g.get("type") == "FACT"]

        agent_states = {
            s: sum(1 for w in self.workers if w.state == s)
            for s in [
                AgentState.IDLE,
                AgentState.THINKING,
                AgentState.EXECUTING,
                AgentState.CLAIMING,
                AgentState.VERIFYING,
            ]
        }

        total_cmd_runs = sum(w.commands_run for w in self.workers)

        return {
            "workers_count": self.workers_count,
            "max_active_concurrency": self.max_active,
            "elapsed_seconds": round(elapsed, 2),
            "total_turns": self.total_turns_completed,
            "board_total_entries": len(all_gists),
            "board_live_entries": len(live_gists),
            "patches_verified": len(patches),
            "active_claims": len(claims),
            "fails_recorded": len(fails),
            "facts_established": len(facts),
            "sandbox_commands_executed": total_cmd_runs,
            "agent_states": agent_states,
        }

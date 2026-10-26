#!/usr/bin/env python3
"""
Agensh (Open Source Edition) CLI Launcher.
Scalable Decentralized Multi-Agent Harness (4 to 1024 Agents).
Powered by Gemma 4, E2B Sandboxes, SQLite, and Node.js.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure Agensh package is importable
repo_root = Path(__file__).resolve().parent
if (repo_root / "Agensh-1.0").exists():
    sys.path.insert(0, str(repo_root / "Agensh-1.0"))
sys.path.insert(0, str(repo_root))

from agensh_core.config import AgenshConfig
from agensh_core.orchestrator.engine import ScalableHarness
from agensh_core.web.app import init_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agensh.launcher")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agensh: Scalable Multi-Agent Harness (4 - 1024 Agents) with Gemma 4 & E2B"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=int(os.environ.get("PBENCH_WORKERS", 8)),
        help="Number of collaborating agents (scalable from 4 to 1024, default: 8)",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=int(os.environ.get("MAX_ACTIVE_WORKERS", 32)),
        help="Max active concurrent agent coroutines (default: 32)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.environ.get("LLM_MODEL", "gemma4"),
        help="Model name (e.g. 'gemma4', 'gemma-4-it', 'gemma:latest', default: gemma4)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("LLM_PROVIDER", "ollama"),
        choices=["ollama", "vllm", "openai", "gemini", "mock"],
        help="LLM provider runtime (default: ollama)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
        help="LLM API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("LLM_API_KEY", os.environ.get("GEMINI_API_KEY", "dummy")),
        help="LLM API key (if required)",
    )
    parser.add_argument(
        "--e2b-api-key",
        type=str,
        default=os.environ.get("E2B_API_KEY", None),
        help="E2B Cloud API key (if omitted, high-speed local directory sandbox is used)",
    )
    parser.add_argument(
        "--use-local-sandbox",
        action="store_true",
        default=False,
        help="Force using local isolated sandbox workspace instead of E2B cloud",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.environ.get("SQLITE_DB_PATH", "agensh.db"),
        help="SQLite database path (default: agensh.db)",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=5,
        help="Number of cooperation rounds to run (default: 5)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=0.0,
        help="Run continuous harness for D seconds instead of fixed rounds",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=os.environ.get("GEMMA_MOCK", "0") in ("1", "true"),
        help="Simulate Gemma 4 agent responses (ideal for testing 4-1024 agents locally)",
    )
    parser.add_argument(
        "--serve-web",
        action="store_true",
        default=False,
        help="Start and serve the FastAPI REST server & real-time Web Dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8680,
        help="Web dashboard port (default: 8680)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="pbench-pandoc",
        help="Task identifier (default: pbench-pandoc)",
    )

    return parser.parse_args()


def start_web_server(app, host: str, port: int):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


async def main():
    args = parse_args()

    # Validate workers bounds
    workers = max(4, min(1024, args.workers))
    concurrency = min(workers, max(1, args.concurrency))

    # For local Ollama inference, optimize concurrency if left at default
    if args.provider == "ollama" and not args.mock_llm and concurrency > 4 and args.concurrency == 32:
        concurrency = min(workers, 4)

    # Auto-detect Ollama model tags if model was given as generic 'gemma4'
    selected_model = args.model
    if args.provider == "ollama" and not args.mock_llm:
        try:
            import httpx
            r = httpx.get(f"{args.base_url.rstrip('/')}/api/tags", timeout=3.0)
            if r.status_code == 200:
                tags = [m.get("name", "") for m in r.json().get("models", [])]
                if selected_model not in tags:
                    matched = [t for t in tags if t.startswith(f"{selected_model}:") or selected_model in t]
                    if matched:
                        selected_model = matched[0]
                        print(f"[*] Auto-detected Ollama tag: '{selected_model}'")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("[+] AGENSH (OPEN SOURCE EDITION) MULTI-AGENT HARNESS")
    print(f"    Agents Fleet:        {workers} agents (Scalable 4 - 1024)")
    print(f"    Active Concurrency:  {concurrency} concurrent coroutines")
    print(f"    Model Architecture:  Google Gemma 4 ({selected_model} via {args.provider})")
    print(f"    Execution Sandbox:   {'E2B Cloud' if args.e2b_api_key and not args.use_local_sandbox else 'Local Isolated Sandbox'}")
    print(f"    Database Engine:     SQLite (WAL mode, zero-maintenance)")
    print(f"    Task ID:             {args.task_id}")
    if args.mock_llm:
        print("    LLM Mode:            MOCK / SIMULATED (Dry run / scale verification)")
    print("=" * 70 + "\n")

    config = AgenshConfig(
        workers=workers,
        max_active_workers=concurrency,
        llm_provider=args.provider,
        llm_model=selected_model,
        llm_base_url=args.base_url,
        llm_api_key=args.api_key,
        mock_llm=args.mock_llm,
        e2b_api_key=args.e2b_api_key,
        use_local_sandbox=args.use_local_sandbox,
        sqlite_db_path=Path(args.db),
        task_id=args.task_id,
    )

    harness = ScalableHarness(config)
    await harness.initialize()

    # Optional background Web Dashboard server
    if args.serve_web:
        web_app = init_app(harness.store, harness.board_service)
        web_thread = threading.Thread(
            target=start_web_server,
            args=(web_app, "0.0.0.0", args.port),
            daemon=True,
        )
        web_thread.start()
        print(f"[*] Live Visualizer Dashboard: http://localhost:{args.port}/\n")

    def print_turn_event(res: dict):
        agent = res.get("agent", "")
        turn = res.get("turn", 0)
        tools = [tr["tool"] for tr in res.get("tool_results", [])]
        tools_str = ", ".join(tools) if tools else "idle"
        print(f"  [{agent}] Turn {turn}: {tools_str}")

    try:
        if args.duration > 0:
            print(f"[*] Running continuous async harness for {args.duration}s...")
            summary = await harness.run_continuous(args.duration)
        else:
            print(f"[*] Running {args.rounds} rounds across all {workers} agents...")
            summary = await harness.run_rounds(rounds=args.rounds, on_turn_done=print_turn_event)

        print("\n" + "=" * 70)
        print("[*] AGENSH MULTI-AGENT RUN SUMMARY")
        print("=" * 70)
        print(f"  Total Agents:                  {summary['workers_count']}")
        print(f"  Active Concurrency:            {summary['max_active_concurrency']}")
        print(f"  Elapsed Time:                  {summary['elapsed_seconds']} seconds")
        print(f"  Total Agent Turns:             {summary['total_turns']}")
        print(f"  Total Shared Board Entries:    {summary['board_total_entries']}")
        print(f"  Active Claims:                 {summary['active_claims']}")
        print(f"  Verified Patches:              {summary['patches_verified']}")
        print(f"  Failures Filtered:             {summary['fails_recorded']}")
        print(f"  Facts Established:             {summary['facts_established']}")
        print(f"  E2B Sandbox Commands Run:      {summary['sandbox_commands_executed']}")
        print("=" * 70 + "\n")

        if args.serve_web:
            print(f"[*] Dashboard remains active at http://localhost:{args.port}/. Press Ctrl+C to terminate.")
            while True:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping Agensh harness...")
        harness.stop()
    finally:
        await harness.store.close()


if __name__ == "__main__":
    asyncio.run(main())

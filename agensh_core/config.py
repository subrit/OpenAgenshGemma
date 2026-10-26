"""
Configuration module for Agensh (Open Source Edition).
Supports scaling from 4 to 1024 agents, Gemma 4 LLM, E2B sandbox, and SQLite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AgenshConfig:
    # Agent scaling
    workers: int = 8                    # Total number of agents (4 to 1024)
    max_active_workers: int = 32        # Concurrency limit for active agent coroutines
    turn_timeout_seconds: float = 180.0 # Timeout per agent turn
    max_turns_per_agent: int = 50       # Maximum iterations per agent
    simulation_duration_seconds: float = 3600.0  # Run time budget

    # LLM Settings (Gemma 4)
    llm_provider: str = "ollama"        # 'ollama', 'vllm', 'openai', 'gemini', 'mock'
    llm_model: str = "gemma4"           # 'gemma4', 'gemma-4-it', 'gemma:latest', etc.
    llm_base_url: str = "http://localhost:11434" # Inference base URL
    llm_api_key: str = "dummy"          # API key for hosted providers
    llm_max_concurrency: int = 16       # Concurrency limiter for LLM calls
    llm_rpm_limit: int = 600            # Requests per minute rate limit
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    mock_llm: bool = False              # Set True for dry-runs / synthetic tests

    # E2B Sandbox Settings
    e2b_api_key: Optional[str] = None   # E2B cloud API key (None = use local isolated sandbox)
    e2b_template: str = "code-interpreter-v1"
    e2b_timeout: int = 300              # Sandbox timeout in seconds
    use_local_sandbox: bool = False     # Force local sandbox fallback
    sandbox_base_dir: Path = field(default_factory=lambda: Path(".agensh_sandboxes").absolute())

    # SQLite Database Settings
    sqlite_db_path: Path = field(default_factory=lambda: Path("agensh.db").absolute())
    board_claim_ttl_seconds: float = 3600.0  # Claims expire after 1 hour by default
    board_max_entries: int = 2000

    # Networking & Service Settings
    board_port: int = 8660
    dashboard_port: int = 8680
    host: str = "0.0.0.0"

    # Task Settings
    task_id: str = "pbench-task-1"
    task_name: str = "ProgramBench Task"
    task_brief: str = (
        "Rebuild reference programs from behavioral observations and documentation. "
        "Agents collaborate via shared context board, claim sub-tasks, execute and verify "
        "in E2B sandboxes, and publish verified patch summaries."
    )

    @classmethod
    def from_env(cls) -> AgenshConfig:
        workers = int(os.environ.get("PBENCH_WORKERS", os.environ.get("AGENTS_COUNT", "8")))
        workers = max(4, min(1024, workers))

        max_active = int(os.environ.get("MAX_ACTIVE_WORKERS", str(min(workers, 32))))

        e2b_key = os.environ.get("E2B_API_KEY")
        use_local = os.environ.get("USE_LOCAL_SANDBOX", "0").lower() in ("1", "true", "yes") or not e2b_key

        mock_llm = os.environ.get("GEMMA_MOCK", "0").lower() in ("1", "true", "yes")

        db_path = Path(os.environ.get("SQLITE_DB_PATH", "agensh.db")).resolve()

        return cls(
            workers=workers,
            max_active_workers=max_active,
            llm_provider=os.environ.get("LLM_PROVIDER", "ollama"),
            llm_model=os.environ.get("LLM_MODEL", "gemma4"),
            llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
            llm_api_key=os.environ.get("LLM_API_KEY", os.environ.get("GEMINI_API_KEY", "dummy")),
            mock_llm=mock_llm,
            e2b_api_key=e2b_key,
            use_local_sandbox=use_local,
            sqlite_db_path=db_path,
            board_claim_ttl_seconds=float(os.environ.get("BOARD_CLAIM_TTL_SECONDS", "3600")),
            task_id=os.environ.get("TASK_ID", "pbench-task-1"),
        )

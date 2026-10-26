"""
E2B Sandbox Integration for Agensh.
Provides isolated cloud execution via E2B Code Interpreter SDK,
with a robust local directory/subprocess fallback when running offline.
Eliminates the need for local Docker root containers and cgroup slices.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("agensh.e2b")

# Try importing official E2B SDK
try:
    from e2b_code_interpreter import Sandbox as E2BCodeSandbox
    HAS_E2B_SDK = True
except ImportError:
    try:
        from e2b import Sandbox as E2BCodeSandbox
        HAS_E2B_SDK = True
    except ImportError:
        HAS_E2B_SDK = False


class BaseSandbox:
    """Abstract interface for agent sandboxes."""

    async def exec_cmd(
        self, command: str, cwd: Optional[str] = None, timeout: float = 60.0
    ) -> Tuple[int, str, str]:
        raise NotImplementedError

    async def read_file(self, path: str) -> str:
        raise NotImplementedError

    async def write_file(self, path: str, content: str) -> bool:
        raise NotImplementedError

    async def run_code(
        self, code: str, language: str = "python"
    ) -> Tuple[str, str, Any]:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class CloudE2BSandbox(BaseSandbox):
    """Sandbox backed by E2B Cloud micro-VMs."""

    def __init__(self, api_key: str, template: str = "code-interpreter-v1"):
        self.api_key = api_key
        self.template = template
        self._sandbox: Optional[Any] = None

    async def _ensure_sandbox(self) -> Any:
        if self._sandbox is None:
            loop = asyncio.get_running_loop()
            self._sandbox = await loop.run_in_executor(
                None, lambda: E2BCodeSandbox(api_key=self.api_key)
            )
        return self._sandbox

    async def exec_cmd(
        self, command: str, cwd: Optional[str] = None, timeout: float = 60.0
    ) -> Tuple[int, str, str]:
        sb = await self._ensure_sandbox()
        loop = asyncio.get_running_loop()

        def _run():
            res = sb.commands.run(command, cwd=cwd, timeout=timeout)
            return res.exit_code, res.stdout, res.stderr

        return await loop.run_in_executor(None, _run)

    async def read_file(self, path: str) -> str:
        sb = await self._ensure_sandbox()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: sb.files.read(path))

    async def write_file(self, path: str, content: str) -> bool:
        sb = await self._ensure_sandbox()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: sb.files.write(path, content))
        return True

    async def run_code(
        self, code: str, language: str = "python"
    ) -> Tuple[str, str, Any]:
        sb = await self._ensure_sandbox()
        loop = asyncio.get_running_loop()

        def _exec():
            res = sb.run_code(code, language=language)
            stdout = "\n".join(res.logs.stdout) if hasattr(res, "logs") else ""
            stderr = "\n".join(res.logs.stderr) if hasattr(res, "logs") else ""
            return stdout, stderr, res.results

        return await loop.run_in_executor(None, _exec)

    async def close(self) -> None:
        if self._sandbox is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._sandbox.close())
            self._sandbox = None


class LocalSandbox(BaseSandbox):
    """
    Lightweight, isolated directory sandbox for local offline development,
    CI/CD, or environments without an E2B cloud subscription.
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel_path: str) -> Path:
        target = (self.workspace_root / rel_path).resolve()
        if not str(target).startswith(str(self.workspace_root)):
            raise ValueError(f"Path escape attempt: {rel_path}")
        return target

    async def exec_cmd(
        self, command: str, cwd: Optional[str] = None, timeout: float = 60.0
    ) -> Tuple[int, str, str]:
        import sys
        work_dir = self._resolve(cwd) if cwd else self.workspace_root
        work_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()

        def _run():
            try:
                # Replace python with sys.executable for reliable execution on Windows
                cmd = command
                if cmd.startswith("python "):
                    cmd = f'"{sys.executable}" ' + cmd[7:]
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired:
                return -1, "", f"Command timed out after {timeout} seconds"
            except Exception as e:
                return -1, "", str(e)

        return await loop.run_in_executor(None, _run)

    async def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return target.read_text(encoding="utf-8", errors="replace")

    async def write_file(self, path: str, content: str) -> bool:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return True

    async def run_code(
        self, code: str, language: str = "python"
    ) -> Tuple[str, str, Any]:
        temp_file = self.workspace_root / f"_tmp_{int(time.time()*1000)}.py"
        temp_file.write_text(code, encoding="utf-8")
        try:
            code_ret, stdout, stderr = await self.exec_cmd(f"python {temp_file.name}")
            return stdout, stderr, {"exit_code": code_ret}
        finally:
            if temp_file.exists():
                temp_file.unlink()

    async def close(self) -> None:
        pass


class E2BSandboxManager:
    """Manages lifecycle and pooling of sandboxes for 4 to 1024 agents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_dir: Path | str = ".agensh_sandboxes",
        use_local: bool = False,
        pool_sandboxes: bool = False,
        max_sandboxes: int = 32,
    ):
        self.api_key = api_key
        self.base_dir = Path(base_dir).resolve()
        self.use_local = use_local or not api_key or not HAS_E2B_SDK
        self.pool_sandboxes = pool_sandboxes
        self.max_sandboxes = max_sandboxes
        self._sandboxes: Dict[str, BaseSandbox] = {}
        self._lock = asyncio.Lock()

        if self.use_local:
            logger.info("Using high-speed Local Directory Sandboxes (isolated workspace per agent).")
        else:
            logger.info("Using E2B Cloud Sandboxes via official E2B SDK.")

    async def get_sandbox(self, agent_name: str) -> BaseSandbox:
        """Get or initialize a sandbox for the given agent."""
        async with self._lock:
            # In pooled mode, map agents into a pool of sandboxes to conserve resources
            key = (
                f"pool_{hash(agent_name) % self.max_sandboxes}"
                if self.pool_sandboxes
                else agent_name
            )

            if key not in self._sandboxes:
                if self.use_local or not self.api_key or not HAS_E2B_SDK:
                    agent_dir = self.base_dir / agent_name
                    self._sandboxes[key] = LocalSandbox(agent_dir)
                else:
                    try:
                        self._sandboxes[key] = CloudE2BSandbox(api_key=self.api_key)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to create Cloud E2B sandbox ({exc}), falling back to Local Sandbox."
                        )
                        agent_dir = self.base_dir / agent_name
                        self._sandboxes[key] = LocalSandbox(agent_dir)

            return self._sandboxes[key]

    async def cleanup_all(self) -> None:
        async with self._lock:
            for sb in self._sandboxes.values():
                try:
                    await sb.close()
                except Exception:
                    pass
            self._sandboxes.clear()

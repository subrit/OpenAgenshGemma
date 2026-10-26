"""
Autonomous Agent Worker for Agensh.
Implements the decentralized multi-agent cooperation loop:
Gather Context -> Claim Sub-Task -> E2B Action -> Verify -> Share Lessons.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..board.service import BoardService
from ..db.sqlite_store import SQLiteStore
from ..llm.gemma4_client import Gemma4Client
from ..llm.prompts import AGENT_SYSTEM_PROMPT, TASK_BRIEF_TEMPLATE
from ..sandbox.e2b_manager import E2BSandboxManager

logger = logging.getLogger("agensh.worker")


class AgentState:
    IDLE = "IDLE"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING_E2B"
    CLAIMING = "CLAIMING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"


class AgentWorker:
    """An autonomous worker operating inside the decentralized Agensh harness."""

    def __init__(
        self,
        agent_id: str,
        total_workers: int,
        task_id: str,
        task_brief: str,
        board_service: BoardService,
        llm_client: Gemma4Client,
        sandbox_manager: E2BSandboxManager,
        store: SQLiteStore,
        board_ids: Optional[List[str]] = None,
    ):
        self.agent_id = agent_id
        self.total_workers = total_workers
        self.task_id = task_id
        self.task_brief = task_brief
        self.board_service = board_service
        self.llm_client = llm_client
        self.sandbox_manager = sandbox_manager
        self.store = store
        self.board_ids = board_ids or ["pbench-top"]

        self.state = AgentState.IDLE
        self.turn_count = 0
        self.active_claim: Optional[str] = None
        self.patches_submitted = 0
        self.commands_run = 0
        self._history: List[Dict[str, Any]] = []

    async def run_turn(self) -> Dict[str, Any]:
        """Execute one complete cooperation cycle turn."""
        self.turn_count += 1
        turn_start = time.time()
        self.state = AgentState.THINKING

        try:
            # 1. Fetch unread board updates & pending mid-turn notices
            snapshots = await self.board_service.get_snapshot(self.agent_id, self.board_ids)
            notices = await self.board_service.get_pending_notices(self.agent_id, self.board_ids)

            # 2. Render Prompt & Shared Context
            prompt_context = self.board_service.get_board(self.board_ids[0]).render_prompt_context()

            notices_text = ""
            ack_ids = []
            if notices:
                notice_lines = ["==== URGENT NOTICES & MESSAGES ===="]
                for n in notices:
                    ack_ids.append(n["event_id"])
                    if n.get("kind") == "direct_message":
                        notice_lines.append(f"[DM from {n['from_agent']}]: {n['content']}")
                    else:
                        notice_lines.append(
                            f"[New Peer Note {n.get('label')} by {n.get('from_agent')}]: "
                            f"{n.get('type')} - {n.get('content')}"
                        )
                notice_lines.append("===================================")
                notices_text = "\n" + "\n".join(notice_lines)

            system_content = AGENT_SYSTEM_PROMPT.format(
                agent_name=self.agent_id, total_workers=self.total_workers
            )
            mission_content = TASK_BRIEF_TEMPLATE.format(
                task_id=self.task_id, task_brief=self.task_brief
            )

            user_turn_prompt = (
                f"{mission_content}\n\n"
                f"{prompt_context}\n"
                f"{notices_text}\n\n"
                f"Turn {self.turn_count}. Decide what to observe, claim, test in E2B sandbox, or publish."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_turn_prompt},
            ]

            # 3. Call Gemma 4 LLM
            llm_result = await self.llm_client.generate_turn(
                agent_name=self.agent_id,
                messages=messages,
                turn_number=self.turn_count,
                context_hints={
                    "active_claim": self.active_claim,
                    "unclaimed_tasks": ["parser", "ast", "evaluator", "serializer", "codegen"],
                },
            )

            thought = llm_result.get("content", "")
            tool_calls = llm_result.get("tool_calls", [])

            # 4. Execute tool calls
            tool_results = []
            for tc in tool_calls:
                fn_name = tc.get("name")
                args = tc.get("arguments", {})
                res = await self._dispatch_tool(fn_name, args)
                tool_results.append({"tool": fn_name, "args": args, "result": res})

            # 5. Acknowledge notices
            if ack_ids:
                await self.board_service.acknowledge_notices(self.agent_id, ack_ids)

            duration = time.time() - turn_start
            self.state = AgentState.IDLE

            return {
                "agent": self.agent_id,
                "turn": self.turn_count,
                "thought": thought,
                "tool_results": tool_results,
                "duration_ms": duration * 1000,
            }

        except Exception as exc:
            self.state = AgentState.IDLE
            logger.error(f"[{self.agent_id}] Turn {self.turn_count} error: {exc}", exc_info=True)
            return {
                "agent": self.agent_id,
                "turn": self.turn_count,
                "error": str(exc),
            }

    async def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Route tool invocation to BoardService, E2B Sandbox, or SQLiteStore."""
        sandbox = await self.sandbox_manager.get_sandbox(self.agent_id)

        if name == "board_write":
            self.state = AgentState.CLAIMING if args.get("type") == "CLAIM" else AgentState.VERIFYING
            res = await self.board_service.post_note(
                board_id=self.board_ids[0],
                thread_id=self.agent_id,
                note_type=args["type"],
                content=args["content"],
                delegation_id=args.get("delegation_id"),
                ttl_seconds=args.get("ttl_seconds"),
                detail=args.get("detail"),
            )
            if args.get("type") == "CLAIM":
                self.active_claim = args.get("delegation_id")
            elif args.get("type") == "PATCH_SUMMARY":
                self.patches_submitted += 1
            return res

        elif name == "board_read":
            return await self.board_service.read_board(args.get("board_id", self.board_ids[0]))

        elif name == "board_grep":
            items, total = await self.board_service.grep_board(
                board_id=self.board_ids[0], query_expr=args.get("query", "")
            )
            return {"matches": items, "total_count": total}

        elif name == "board_unfold":
            label = args.get("label", "")
            gid = int(label[1:]) if label.startswith("G") and label[1:].isdigit() else 0
            detail = await self.board_service.unfold(self.board_ids[0], gid)
            return {"label": label, "detail": detail or "(no detail available)"}

        elif name == "e2b_exec":
            self.state = AgentState.EXECUTING
            self.commands_run += 1
            start = time.time()
            code, stdout, stderr = await sandbox.exec_cmd(
                command=args["command"], cwd=args.get("cwd")
            )
            duration_ms = (time.time() - start) * 1000
            await self.store.log_sandbox(
                agent=self.agent_id,
                command=args["command"],
                exit_code=code,
                stdout=stdout[:2000],
                stderr=stderr[:2000],
                duration_ms=duration_ms,
            )
            return {"exit_code": code, "stdout": stdout, "stderr": stderr}

        elif name == "e2b_read_file":
            try:
                content = await sandbox.read_file(args["path"])
                return {"path": args["path"], "content": content}
            except Exception as e:
                return {"error": str(e)}

        elif name == "e2b_write_file":
            try:
                await sandbox.write_file(args["path"], args["content"])
                return {"path": args["path"], "status": "written"}
            except Exception as e:
                return {"error": str(e)}

        elif name == "e2b_run_code":
            self.state = AgentState.EXECUTING
            stdout, stderr, results = await sandbox.run_code(
                args["code"], language=args.get("language", "python")
            )
            return {"stdout": stdout, "stderr": stderr, "results": results}

        elif name == "send_dm":
            msg_id = await self.store.send_message(
                from_agent=self.agent_id,
                to_agent=args["to_agent"],
                channel="direct",
                content=args["content"],
                msg_type="collision" if "collision" in args["content"].lower() else "chat",
            )
            return {"message_id": msg_id, "status": "sent"}

        elif name == "send_channel":
            msg_id = await self.store.send_message(
                from_agent=self.agent_id,
                to_agent=None,
                channel=args.get("channel", "general"),
                content=args["content"],
                msg_type="chat",
            )
            return {"message_id": msg_id, "status": "broadcast"}

        return {"error": f"Unknown tool '{name}'"}

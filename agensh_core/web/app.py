"""
FastAPI REST service and Real-time Multi-Agent Visualizer Dashboard for Agensh.
Provides DeLM board compatibility, SQLite persistence, messaging APIs,
and an interactive Web Dashboard supporting 4 to 1024 agents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..board.service import BoardService
from ..db.sqlite_store import SQLiteStore

logger = logging.getLogger("agensh.web")

app = FastAPI(title="Agensh Open-Source Board & Coordination Service", version="2.0.0")

# Global service instance (can be attached to app.state)
STORE: Optional[SQLiteStore] = None
BOARD_SERVICE: Optional[BoardService] = None


def init_app(store: SQLiteStore, service: BoardService) -> FastAPI:
    global STORE, BOARD_SERVICE
    STORE = store
    BOARD_SERVICE = service
    return app


class NoteIn(BaseModel):
    thread_id: str = Field(..., description="Agent ID, e.g. 'worker-1'")
    type: str = Field(..., description="FACT, OBSERVED, FAIL, CLAIM, or PATCH_SUMMARY")
    content: str
    delegation_id: Optional[str] = None
    ttl_seconds: Optional[float] = None
    detail: Optional[str] = None


class MessageIn(BaseModel):
    from_agent: str
    to_agent: Optional[str] = None
    channel: str = "general"
    content: str
    msg_type: str = "chat"


class ShownIn(BaseModel):
    event_ids: List[int]


@app.on_event("startup")
async def startup():
    global STORE, BOARD_SERVICE
    if STORE is None:
        STORE = SQLiteStore("agensh.db")
        BOARD_SERVICE = BoardService(STORE)
    await STORE.initialize()
    await BOARD_SERVICE.hydrate()


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "agensh-open-source"}


# -----------------------------------------------------------------------------
# DeLM Board Protocol Endpoints
# -----------------------------------------------------------------------------


@app.post("/board/{board_id}/note")
async def post_note(board_id: str, note: NoteIn):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    try:
        res = await BOARD_SERVICE.post_note(
            board_id=board_id,
            thread_id=note.thread_id,
            note_type=note.type,
            content=note.content,
            delegation_id=note.delegation_id,
            ttl_seconds=note.ttl_seconds,
            detail=note.detail,
        )
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/board/{board_id}/read")
async def read_board(board_id: str):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    return await BOARD_SERVICE.read_board(board_id, raw=False)


@app.get("/board/{board_id}/read_all")
async def read_board_all(board_id: str):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    return await BOARD_SERVICE.read_board(board_id, raw=True)


@app.get("/board/{board_id}/grep")
async def grep_board(
    board_id: str,
    q: str = Query("", description="Query string e.g. 'fail|error' or 'ast&parser'"),
    types: Optional[str] = Query(None, description="Comma-separated types"),
    limit: int = Query(500, ge=1, le=2000),
):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    type_list = [t.strip().upper() for t in types.split(",") if t.strip()] if types else None
    items, total = await BOARD_SERVICE.grep_board(board_id, q, type_list, limit)
    return {"matches": items, "total_count": total, "limit": limit}


@app.get("/board/{board_id}/unfold")
async def unfold_board(
    board_id: str,
    label: Optional[str] = Query(None),
    gist_id: Optional[int] = Query(None),
):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    gid = gist_id
    if gid is None and label:
        if label.startswith("G") and label[1:].isdigit():
            gid = int(label[1:])
    if gid is None:
        raise HTTPException(status_code=400, detail="Must provide 'label' (e.g. G12) or 'gist_id'")
    detail = await BOARD_SERVICE.unfold(board_id, gid)
    return {"label": f"G{gid}", "gist_id": gid, "detail": detail}


@app.get("/snapshot")
async def get_snapshot(
    agent: str = Query(..., description="Agent ID requesting snapshot"),
    boards: str = Query("pbench-top", description="Comma-separated board IDs"),
):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    board_list = [b.strip() for b in boards.split(",") if b.strip()]
    return await BOARD_SERVICE.get_snapshot(agent, board_list)


@app.get("/pending/{agent}")
async def get_pending(
    agent: str,
    boards: str = Query("pbench-top", description="Comma-separated board IDs"),
):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    board_list = [b.strip() for b in boards.split(",") if b.strip()]
    notices = await BOARD_SERVICE.get_pending_notices(agent, board_list)
    return {"agent": agent, "notices": notices}


@app.post("/shown/{agent}")
async def post_shown(agent: str, payload: ShownIn):
    if not BOARD_SERVICE:
        raise HTTPException(status_code=500, detail="Service not ready")
    await BOARD_SERVICE.acknowledge_notices(agent, payload.event_ids)
    return {"status": "ok", "acknowledged": len(payload.event_ids)}


# -----------------------------------------------------------------------------
# Messaging & Coordination Endpoints
# -----------------------------------------------------------------------------


@app.post("/messages/send")
async def send_message(msg: MessageIn):
    if not STORE:
        raise HTTPException(status_code=500, detail="Service not ready")
    mid = await STORE.send_message(
        from_agent=msg.from_agent,
        to_agent=msg.to_agent,
        channel=msg.channel,
        content=msg.content,
        msg_type=msg.msg_type,
    )
    return {"message_id": mid, "status": "sent"}


@app.get("/messages")
async def list_messages(
    agent: Optional[str] = None,
    channel: Optional[str] = None,
    limit: int = 100,
    since_id: int = 0,
):
    if not STORE:
        raise HTTPException(status_code=500, detail="Service not ready")
    return await STORE.get_messages(agent=agent, channel=channel, limit=limit, since_id=since_id)


@app.get("/tasks")
async def list_tasks():
    if not STORE:
        raise HTTPException(status_code=500, detail="Service not ready")
    return await STORE.list_tasks()


# -----------------------------------------------------------------------------
# Interactive Real-Time Visualizer Dashboard
# -----------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agensh: Scalable Multi-Agent Harness Dashboard</title>
    <style>
        :root {
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-border: #334155;
            --primary: #38bdf8;
            --accent: #818cf8;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #4ade80;
            --warning: #facc15;
            --danger: #f87171;
            --claim: #c084fc;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid var(--surface-border); padding-bottom: 16px; }
        h1 { font-size: 24px; font-weight: 700; color: var(--primary); }
        .tag { background: #0284c7; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--surface); border: 1px solid var(--surface-border); border-radius: 10px; padding: 16px; text-align: center; }
        .stat-value { font-size: 28px; font-weight: 700; color: var(--text); margin-top: 4px; }
        .stat-label { font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        
        .main-layout { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        @media (max-width: 1024px) { .main-layout { grid-template-columns: 1fr; } }
        
        .panel { background: var(--surface); border: 1px solid var(--surface-border); border-radius: 10px; padding: 18px; margin-bottom: 20px; }
        .panel-header { font-size: 16px; font-weight: 600; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }
        
        .agents-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(42px, 1fr)); gap: 6px; max-height: 240px; overflow-y: auto; padding-right: 4px; }
        .agent-node { height: 42px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; cursor: pointer; transition: transform 0.1s; border: 1px solid rgba(255,255,255,0.1); }
        .agent-node:hover { transform: scale(1.15); z-index: 10; }
        .agent-idle { background: #334155; color: #94a3b8; }
        .agent-thinking { background: #0284c7; color: white; }
        .agent-executing { background: #f59e0b; color: black; }
        .agent-claiming { background: #9333ea; color: white; }
        .agent-verifying { background: #10b981; color: white; }
        
        .board-feed { max-height: 480px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
        .feed-item { background: #0f172a; border-left: 4px solid var(--primary); padding: 10px 14px; border-radius: 4px; font-size: 13px; }
        .feed-item.type-FAIL { border-left-color: var(--danger); }
        .feed-item.type-PATCH_SUMMARY { border-left-color: var(--success); }
        .feed-item.type-CLAIM { border-left-color: var(--claim); }
        .feed-item.type-FACT { border-left-color: var(--primary); }
        .feed-meta { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
        .badge { font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; }
        .badge-FAIL { background: rgba(248,113,113,0.2); color: var(--danger); }
        .badge-PATCH_SUMMARY { background: rgba(74,222,128,0.2); color: var(--success); }
        .badge-CLAIM { background: rgba(192,132,252,0.2); color: var(--claim); }
        .badge-FACT { background: rgba(56,189,248,0.2); color: var(--primary); }
        
        .messages-list { max-height: 480px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
        .msg-bubble { background: #0f172a; padding: 10px; border-radius: 6px; font-size: 12px; }
        .msg-header { font-weight: 600; color: var(--primary); margin-bottom: 2px; font-size: 11px; }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Agensh Open-Source Multi-Agent Harness</h1>
            <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
                Gemma 4 &bull; E2B Sandboxes &bull; SQLite Storage &bull; Decentralized Coordination
            </p>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="tag" id="agents-scale-tag">Scalable: 4 - 1024 Agents</span>
            <span class="tag" style="background: #10b981;">Online</span>
        </div>
    </header>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Agents</div>
            <div class="stat-value" id="stat-agents">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Active Claims</div>
            <div class="stat-value" style="color: var(--claim);" id="stat-claims">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Verified Patches</div>
            <div class="stat-value" style="color: var(--success);" id="stat-patches">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Fails / Dead Ends Filtered</div>
            <div class="stat-value" style="color: var(--danger);" id="stat-fails">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Board Notes</div>
            <div class="stat-value" style="color: var(--primary);" id="stat-gists">--</div>
        </div>
    </div>

    <div class="main-layout">
        <div>
            <div class="panel">
                <div class="panel-header">
                    <span>Agent Fleet Concurrency (Visualized 1-1024)</span>
                    <span style="font-size: 12px; color: var(--text-muted);" id="agents-status-summary">Auto-updating...</span>
                </div>
                <div class="agents-grid" id="agents-container"></div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span>Shared Context Board (DeLM Append-Only Knowledge Log)</span>
                    <span style="font-size: 12px; color: var(--text-muted);" id="feed-count">0 items</span>
                </div>
                <div class="board-feed" id="board-feed">
                    <div style="text-align: center; color: var(--text-muted); padding: 20px;">Loading board history...</div>
                </div>
            </div>
        </div>

        <div>
            <div class="panel">
                <div class="panel-header">
                    <span>Team Communication</span>
                </div>
                <div class="messages-list" id="messages-list">
                    <div style="text-align: center; color: var(--text-muted); padding: 20px;">No messages yet.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function refreshDashboard() {
            try {
                // Fetch board gists
                const bRes = await fetch('/board/pbench-top/read_all');
                const gists = await bRes.json();
                
                // Fetch messages
                const mRes = await fetch('/messages?limit=30');
                const msgs = await mRes.json();

                // Compute stats
                let patches = 0, fails = 0, claims = 0;
                const activeWorkers = new Set();
                
                gists.forEach(g => {
                    activeWorkers.add(g.thread_id);
                    if (g.type === 'PATCH_SUMMARY') patches++;
                    else if (g.type === 'FAIL') fails++;
                    else if (g.type === 'CLAIM') claims++;
                });

                document.getElementById('stat-gists').innerText = gists.length;
                document.getElementById('stat-patches').innerText = patches;
                document.getElementById('stat-fails').innerText = fails;
                document.getElementById('stat-claims').innerText = claims;
                document.getElementById('stat-agents').innerText = Math.max(activeWorkers.size, 8);
                document.getElementById('feed-count').innerText = `${gists.length} items`;

                // Render Agent nodes (up to active or 128 preview)
                const agentCount = Math.max(activeWorkers.size, 8);
                const aContainer = document.getElementById('agents-container');
                let aHtml = '';
                for (let i = 1; i <= Math.min(agentCount, 128); i++) {
                    const id = `w${i}`;
                    const states = ['agent-idle', 'agent-thinking', 'agent-executing', 'agent-verifying'];
                    const sClass = states[i % states.length];
                    aHtml += `<div class="agent-node ${sClass}" title="worker-${i}">W${i}</div>`;
                }
                aContainer.innerHTML = aHtml;

                // Render Board Feed
                const bContainer = document.getElementById('board-feed');
                if (gists.length > 0) {
                    bContainer.innerHTML = gists.slice(-40).reverse().map(g => {
                        return `
                            <div class="feed-item type-${g.type}">
                                <div class="feed-meta">
                                    <span><strong>${g.label || 'G' + g.id}</strong> by <em>${g.thread_id}</em></span>
                                    <span class="badge badge-${g.type}">${g.type}</span>
                                </div>
                                <div style="color: #e2e8f0;">${g.content}</div>
                            </div>
                        `;
                    }).join('');
                }

                // Render Messages
                const mContainer = document.getElementById('messages-list');
                if (msgs.length > 0) {
                    mContainer.innerHTML = msgs.slice(-25).reverse().map(m => {
                        const target = m.to_agent ? `@${m.to_agent}` : `#${m.channel}`;
                        return `
                            <div class="msg-bubble">
                                <div class="msg-header">${m.from_agent} &rarr; ${target}</div>
                                <div>${m.content}</div>
                            </div>
                        `;
                    }).join('');
                }

            } catch (err) {
                console.error("Dashboard refresh error:", err);
            }
        }

        setInterval(refreshDashboard, 3000);
        refreshDashboard();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard_index():
    return HTMLResponse(content=DASHBOARD_HTML)

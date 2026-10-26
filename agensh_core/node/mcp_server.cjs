#!/usr/bin/env node
/**
 * Agensh Open-Source MCP Server (Model Context Protocol).
 * Exposes board, SQLite messaging, and E2B sandbox execution tools over stdio JSON-RPC.
 * Compatible with Gemma 4 agents, Claude Desktop, Cursor, and any MCP client.
 */

"use strict";

const readline = require("node:readline");
const http = require("node:http");
const https = require("node:https");
const { URL } = require("node:url");

const BOARD_URL = process.env.BOARD_SERVICE_URL || "http://127.0.0.1:8660";
const AGENT = process.env.AGENT_NAME || "worker-1";
const BOARD_ID = process.env.BOARD_ID || "pbench-top";

function log(...args) {
    process.stderr.write(`[agensh-mcp:${AGENT}] ` + args.map(String).join(" ") + "\n");
}

function httpRequest(method, path, body) {
    return new Promise((resolve, reject) => {
        const u = new URL(path, BOARD_URL);
        const client = u.protocol === "https:" ? https : http;
        const payload = body ? Buffer.from(JSON.stringify(body)) : null;
        const req = client.request(
            {
                method,
                hostname: u.hostname,
                port: u.port,
                path: u.pathname + u.search,
                headers: {
                    "Content-Type": "application/json",
                    ...(payload ? { "Content-Length": String(payload.length) } : {}),
                },
            },
            (res) => {
                const chunks = [];
                res.on("data", (c) => chunks.push(c));
                res.on("end", () => {
                    const text = Buffer.concat(chunks).toString("utf8");
                    let json = null;
                    try {
                        json = text ? JSON.parse(text) : null;
                    } catch (_) {}
                    resolve({ status: res.statusCode || 0, text, json });
                });
            }
        );
        req.on("error", reject);
        if (payload) req.write(payload);
        req.end();
    });
}

const TOOLS = [
    {
        name: "board_write",
        description: "Publish a note to the shared context board (FACT, OBSERVED, FAIL, CLAIM, PATCH_SUMMARY).",
        inputSchema: {
            type: "object",
            properties: {
                type: { type: "string", enum: ["FACT", "OBSERVED", "FAIL", "CLAIM", "PATCH_SUMMARY"] },
                content: { type: "string", description: "Short summary of the note." },
                delegation_id: { type: "string", description: "Task or component identifier." },
                detail: { type: "string", description: "Full traceback or detailed note." },
                ttl_seconds: { type: "number", description: "TTL in seconds for CLAIM." },
            },
            required: ["type", "content"],
        },
    },
    {
        name: "board_read",
        description: "Read the recent live entries on the shared context board.",
        inputSchema: {
            type: "object",
            properties: {
                board_id: { type: "string", default: BOARD_ID },
            },
        },
    },
    {
        name: "board_grep",
        description: "Search historical board entries with boolean queries (e.g. 'ast&parser' or 'fail|error').",
        inputSchema: {
            type: "object",
            properties: {
                query: { type: "string", description: "Search query." },
            },
            required: ["query"],
        },
    },
    {
        name: "board_unfold",
        description: "Unfold full detail behind a G<number> label.",
        inputSchema: {
            type: "object",
            properties: {
                label: { type: "string", description: "Label like G12." },
            },
            required: ["label"],
        },
    },
    {
        name: "send_message",
        description: "Send a direct message to a peer agent or broadcast to a channel.",
        inputSchema: {
            type: "object",
            properties: {
                to_agent: { type: "string", description: "Optional peer agent ID for DM." },
                channel: { type: "string", default: "general", description: "Channel name." },
                content: { type: "string", description: "Message content." },
            },
            required: ["content"],
        },
    },
];

async function handleToolCall(name, args) {
    if (name === "board_write") {
        const body = {
            thread_id: AGENT,
            type: args.type,
            content: args.content,
            delegation_id: args.delegation_id || null,
            ttl_seconds: args.ttl_seconds || 3600,
            detail: args.detail || null,
        };
        const res = await httpRequest("POST", `/board/${encodeURIComponent(BOARD_ID)}/note`, body);
        return { content: [{ type: "text", text: JSON.stringify(res.json || res.text) }] };
    }

    if (name === "board_read") {
        const bid = args.board_id || BOARD_ID;
        const res = await httpRequest("GET", `/board/${encodeURIComponent(bid)}/read`);
        return { content: [{ type: "text", text: JSON.stringify(res.json || res.text) }] };
    }

    if (name === "board_grep") {
        const q = encodeURIComponent(args.query || "");
        const res = await httpRequest("GET", `/board/${encodeURIComponent(BOARD_ID)}/grep?q=${q}`);
        return { content: [{ type: "text", text: JSON.stringify(res.json || res.text) }] };
    }

    if (name === "board_unfold") {
        const lab = encodeURIComponent(args.label || "");
        const res = await httpRequest("GET", `/board/${encodeURIComponent(BOARD_ID)}/unfold?label=${lab}`);
        return { content: [{ type: "text", text: JSON.stringify(res.json || res.text) }] };
    }

    if (name === "send_message") {
        const body = {
            from_agent: AGENT,
            to_agent: args.to_agent || null,
            channel: args.channel || "general",
            content: args.content,
        };
        const res = await httpRequest("POST", "/messages/send", body);
        return { content: [{ type: "text", text: JSON.stringify(res.json || res.text) }] };
    }

    throw new Error(`Unknown tool: ${name}`);
}

// JSON-RPC 2.0 stdio loop
const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: false });

rl.on("line", async (line) => {
    line = line.trim();
    if (!line) return;

    let req;
    try {
        req = JSON.parse(line);
    } catch (_) {
        return;
    }

    const { id, method, params } = req;

    if (method === "initialize") {
        const resp = {
            jsonrpc: "2.0",
            id,
            result: {
                protocolVersion: "2024-11-05",
                capabilities: { tools: {} },
                serverInfo: { name: "agensh-mcp-server", version: "2.0.0" },
            },
        };
        process.stdout.write(JSON.stringify(resp) + "\n");
        return;
    }

    if (method === "tools/list") {
        const resp = {
            jsonrpc: "2.0",
            id,
            result: { tools: TOOLS },
        };
        process.stdout.write(JSON.stringify(resp) + "\n");
        return;
    }

    if (method === "tools/call") {
        try {
            const toolName = params.name;
            const toolArgs = params.arguments || {};
            const result = await handleToolCall(toolName, toolArgs);
            const resp = { jsonrpc: "2.0", id, result };
            process.stdout.write(JSON.stringify(resp) + "\n");
        } catch (err) {
            const resp = {
                jsonrpc: "2.0",
                id,
                error: { code: -32000, message: err.message },
            };
            process.stdout.write(JSON.stringify(resp) + "\n");
        }
        return;
    }

    if (id !== undefined) {
        process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id, error: { code: -32601, message: "Method not found" } }) + "\n");
    }
});

log("Agensh Node.js MCP server ready on stdio");

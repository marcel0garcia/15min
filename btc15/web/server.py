"""
FastAPI web server — serves the live dashboard as a browser UI.
Broadcasts engine state to all connected WebSocket clients every second.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

log = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"


class _ConnectionManager:
    def __init__(self):
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)
        log.debug(f"WS client connected ({len(self._clients)} total)")

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)
        log.debug(f"WS client disconnected ({len(self._clients)} remaining)")

    async def broadcast(self, data: dict):
        dead: Set[WebSocket] = set()
        for ws in self._clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._clients -= dead


def create_app(engine) -> FastAPI:
    app = FastAPI(title="15Min BTC Bot", docs_url=None, redoc_url=None)
    manager = _ConnectionManager()

    # ── Static page ────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (_STATIC / "index.html").read_text()

    # ── WebSocket ──────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            # Send current state immediately on connect
            await websocket.send_json(engine.state)
            # Keep connection alive; client sends pings
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    # ── REST API ───────────────────────────────────────────────────────────

    @app.get("/api/state")
    async def get_state():
        return engine.state

    @app.post("/api/toggle-auto-trade")
    async def toggle_auto_trade():
        engine.cfg.strategy.auto_trade = not engine.cfg.strategy.auto_trade
        engine.state["auto_trade"] = engine.cfg.strategy.auto_trade
        return {"auto_trade": engine.cfg.strategy.auto_trade}

    @app.post("/api/toggle-paper")
    async def toggle_paper():
        engine.cfg.strategy.paper_trade = not engine.cfg.strategy.paper_trade
        engine.state["paper_mode"] = engine.cfg.strategy.paper_trade
        return {"paper_trade": engine.cfg.strategy.paper_trade}

    # ── Background broadcaster ─────────────────────────────────────────────

    @app.on_event("startup")
    async def _start_broadcaster():
        async def _broadcast_loop():
            while True:
                try:
                    if manager._clients:
                        await manager.broadcast(engine.state)
                except Exception as e:
                    log.debug(f"Broadcast error: {e}")
                await asyncio.sleep(1.0)

        asyncio.create_task(_broadcast_loop(), name="ws-broadcaster")

    return app

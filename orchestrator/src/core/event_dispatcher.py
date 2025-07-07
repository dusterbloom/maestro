
import asyncio
from fastapi import WebSocket
from typing import Dict, List, Any

class EventDispatcher:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.connections:
            del self.connections[session_id]

    async def dispatch_event(self, session_id: str, event: Dict[str, Any]):
        if session_id in self.connections:
            await self.connections[session_id].send_json(event)

    async def broadcast_event(self, event: Dict[str, Any]):
        for connection in self.connections.values():
            await connection.send_json(event)

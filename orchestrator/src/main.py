import asyncio
import logging
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.state_machine import StateMachine
from core.event_dispatcher import EventDispatcher
from core.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Orchestrator", version="2.0.0")

# Core components
state_machine = StateMachine()
event_dispatcher = EventDispatcher()
session_manager = SessionManager(state_machine, event_dispatcher)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main WebSocket endpoint for all client communication."""
    await session_manager.handle_connect(session_id, websocket)
    try:
        while websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                data = await websocket.receive_json()
                await session_manager.handle_event(session_id, data)
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session: {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop for session {session_id}: {e}")
                await event_dispatcher.dispatch_event(session_id, {"type": "session.error", "data": {"message": str(e)}})
                break
    finally:
        session_manager.handle_disconnect(session_id)
        logger.info(f"Cleaned up session: {session_id}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    active_sessions = len(state_machine._sessions)
    return {"status": "ok", "active_sessions": active_sessions}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Voice Orchestrator starting up...")
    # Service initialization will go here in the next phase

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Voice Orchestrator shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
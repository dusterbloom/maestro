
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.state_machine import StateMachine
from core.event_dispatcher import EventDispatcher
from core.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Orchestrator", version="2.0.0")

# Initialize core components
state_machine = StateMachine()
event_dispatcher = EventDispatcher()
# The SessionManager now initializes all other services internally
session_manager = SessionManager(state_machine, event_dispatcher)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main WebSocket endpoint for all client communication."""
    await session_manager.handle_connect(session_id, websocket)
    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            try:
                # Increase timeout to handle long processing times (TTS generation can take 2-3s)
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                await session_manager.handle_event(session_id, data)
            except asyncio.TimeoutError:
                # Send keepalive ping to prevent connection timeout
                logger.debug(f"Sending keepalive ping to session {session_id}")
                await websocket.ping()
    except WebSocketDisconnect:
        logger.info(f"Client for session {session_id} disconnected.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket loop for session {session_id}: {e}")
    finally:
        session_manager.handle_disconnect(session_id)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "active_sessions": len(state_machine._sessions)
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Voice Orchestrator is starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Voice Orchestrator is shutting down...")

if __name__ == "__main__":
    import uvicorn
    # This is for local testing. In production, Gunicorn/Uvicorn workers are used.
    uvicorn.run(app, host="0.0.0.0", port=8000)

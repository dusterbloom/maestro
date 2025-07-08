
import asyncio
import json
import logging
import websockets
from config import config

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self, session_id: str, event_callback):
        self.session_id = session_id
        self.event_callback = event_callback
        self.websocket: websockets.WebSocketClientProtocol = None
        self.is_connected = False

    async def connect(self):
        try:
            logger.info(f"STTService attempting to connect to {config.WHISPER_URL}")
            self.websocket = await asyncio.wait_for(websockets.connect(config.WHISPER_URL), timeout=10)
            logger.info(f"STTService WebSocket connected for session {self.session_id}")
            
            # WhisperLive specific configuration
            config_msg = {
                "uid": self.session_id,
                "language": "en",
                "task": "transcribe",
                "model": config.STT_MODEL,
                "use_vad": True,
            }
            logger.info(f"STTService sending config: {config_msg}")
            await self.websocket.send(json.dumps(config_msg))
            logger.info(f"STTService config sent for session {self.session_id}")
            
            self.is_connected = True
            asyncio.create_task(self._listen())
            logger.info(f"STTService for session {self.session_id} connected to WhisperLive.")
        except asyncio.TimeoutError:
            logger.error(f"STTService connection timeout for session {self.session_id} to {config.WHISPER_URL}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"STTService for session {self.session_id} failed to connect: {e}")
            logger.error(f"STTService connection details - URL: {config.WHISPER_URL}, Session: {self.session_id}")
            import traceback
            logger.error(f"STTService traceback: {traceback.format_exc()}")
            self.is_connected = False

    async def _listen(self):
        while self.is_connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                logger.info(f"STTService received raw data for session {self.session_id}: {data}")
                
                # Check for segments and extract transcript - only process completed segments
                if data.get("segments"):
                    segments = data.get("segments")
                    logger.info(f"STTService found {len(segments)} segments for session {self.session_id}")
                    
                    # Only process completed segments to avoid endless loops
                    completed_segments = [s for s in segments if s.get('completed', False)]
                    
                    if completed_segments:
                        transcript = " ".join([s['text'] for s in completed_segments]).strip()
                        logger.info(f"STTService extracted transcript from {len(completed_segments)} completed segments for session {self.session_id}: '{transcript}'")
                        
                        if transcript:
                            logger.info(f"STTService sending transcript.final event for session {self.session_id}")
                            # Fire an event back to the SessionManager
                            await self.event_callback({
                                "type": "transcript.final",
                                "data": {"transcript": transcript}
                            })
                        else:
                            logger.warning(f"STTService empty transcript after joining completed segments for session {self.session_id}")
                    else:
                        logger.info(f"STTService no completed segments yet for session {self.session_id} - waiting for speech completion")
                else:
                    logger.info(f"STTService no segments found in data for session {self.session_id}")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"STTService connection closed for session {self.session_id}.")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Error in STTService listener for session {self.session_id}: {e}")

    async def send_audio(self, audio_chunk: bytes):
        if self.is_connected:
            await self.websocket.send(audio_chunk)

    async def close(self):
        if self.is_connected:
            self.is_connected = False
            await self.websocket.close()
            logger.info(f"STTService connection for session {self.session_id} closed.")

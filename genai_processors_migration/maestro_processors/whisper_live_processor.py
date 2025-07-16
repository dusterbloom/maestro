"""
Fixed WhisperProcessor for Maestro GenAI Processors Implementation
Correctly implements WhisperLive protocol with genai-processors
"""

import asyncio
import json
import logging
import base64
import time
import websockets
from typing import AsyncIterable, Optional, Tuple
from urllib.parse import urlparse
from genai_processors import content_api, processor

logger = logging.getLogger(__name__)


class WhisperProcessor(processor.Processor):
    """Fixed WhisperLive processor using genai-processors framework"""

    def __init__(
        self,
        session_id: str,
        whisper_url: str = "ws://whisper-live:9090",
        max_retries: int = 3,
    ):
        self.session_id = session_id
        self.whisper_url = whisper_url
        self.whisper_ws = None
        self.is_connected = False
        self.transcription_queue = asyncio.Queue()
        self.receive_task = None
        self.max_retries = max_retries
        self.whisper_host, self.whisper_port = self.parse_url(whisper_url)
        logger.info(f"WhisperProcessor initialized for {session_id}")

    def parse_url(self, url: str) -> Tuple[str, int]:
        parsed = urlparse(url)
        return (
            parsed.hostname,
            parsed.port or 9090,
        )  # Default to 9090 if no port specified

    async def _connect_whisper(self) -> bool:
        """Connect to WhisperLive with explicit headers to fix handshake issues."""
        for attempt in range(self.max_retries):
            try:
                self.connection_retries = attempt + 1

                # Create WebSocket URL
                whisper_ws_url = f"ws://{self.whisper_host}:{self.whisper_port}"
                logger.info(
                    f"Attempt {attempt + 1}: Connecting to WhisperLive at: {whisper_ws_url}"
                )

                # Define the required handshake headers explicitly to solve "missing Connection header"
                handshake_headers = {"Connection": "Upgrade", "Upgrade": "websocket"}

                # Add the 'extra_headers' argument to the connect call
                self.whisper_ws = await websockets.connect(
                    whisper_ws_url,
                    extra_headers=handshake_headers,
                    ping_interval=None,
                    ping_timeout=None,
                    open_timeout=10,
                    close_timeout=5,
                )

                logger.info(
                    f"WebSocket connected to WhisperLive for session {self.session_id}"
                )

                # Send WhisperLive configuration as first message (required per protocol)
                config_message = {
                    "uid": self.session_id,
                    "language": "en",  # Default to English, update if needed
                    "task": "transcribe",
                    "use_vad": True,
                    "max_clients": 4,
                    "max_connection_time": 3600,
                    "send_last_n_segments": 10,
                    "no_speech_thresh": 0.45,  # Use the value from config
                    "clip_audio": False,
                    "same_output_threshold": 10,
                }

                logger.info(f"Sending config to WhisperLive: {config_message}")
                await self.whisper_ws.send(json.dumps(config_message))

                # WhisperLive starts processing immediately after config
                self.is_connected = True
                logger.info(
                    f"✅ Session {self.session_id}: WhisperLive connected and configured"
                )

                return True

            except websockets.exceptions.InvalidUpgrade as e:
                logger.error(
                    f"❌ WebSocket handshake failed for session {self.session_id} (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"❌ All attempts failed. Check if WhisperLive is running at ws://{self.whisper_host}:{self.whisper_port}"
                    )
                else:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            except ConnectionRefusedError as e:
                logger.error(
                    f"❌ Connection refused for session {self.session_id} (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"❌ WhisperLive not accessible at ws://{self.whisper_host}:{self.whisper_port}"
                    )
                else:
                    await asyncio.sleep(2**attempt)

            except Exception as e:
                logger.error(
                    f"❌ Connection error for session {self.session_id} (attempt {attempt + 1}): {type(e).__name__}: {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"❌ Failed to connect after {self.max_retries} attempts"
                    )
                else:
                    await asyncio.sleep(2**attempt)

        return False

    async def _receive_loop(self):
        """Receive transcriptions from WhisperLive"""
        try:
            while self.whisper_ws and not self.whisper_ws.closed:
                try:
                    message = await self.whisper_ws.recv()

                    # WhisperLive can send different types of messages
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)

                            # Handle different WhisperLive message types
                            if "segments" in data:
                                # Full transcription with segments
                                for segment in data["segments"]:
                                    text = segment.get("text", "").strip()
                                    if text:
                                        await self.transcription_queue.put(
                                            {
                                                "text": text,
                                                "start": segment.get("start"),
                                                "end": segment.get("end"),
                                                "language": segment.get("language"),
                                            }
                                        )

                            elif "text" in data:
                                # Simple transcription
                                text = data["text"].strip()
                                if text:
                                    await self.transcription_queue.put(
                                        {"text": text, "language": data.get("language")}
                                    )

                            elif "message" in data:
                                # Status message
                                logger.debug(f"WhisperLive status: {data['message']}")

                        except json.JSONDecodeError:
                            # Sometimes WhisperLive sends plain text
                            text = message.strip()
                            if text:
                                await self.transcription_queue.put({"text": text})

                except websockets.exceptions.ConnectionClosed:
                    logger.info("WhisperLive connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error in receive loop: {e}")

        finally:
            self.is_connected = False

    async def _disconnect(self):
        """Disconnect from WhisperLive"""
        self.is_connected = False

        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

        if self.whisper_ws:
            try:
                # Send end signal
                await self.whisper_ws.send(json.dumps({"action": "end"}))
            except:
                pass
            await self.whisper_ws.close()
            self.whisper_ws = None

        logger.info("Disconnected from WhisperLive")

    async def call(
        self, input_stream: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Process audio through WhisperLive - fixed implementation"""
        await self._connect_whisper()

        try:
            async for part in input_stream:
                # Check if it's audio
                if (
                    hasattr(part, "bytes")
                    and part.mimetype
                    and "audio" in part.mimetype
                ):
                    audio_data = part.bytes
                elif (
                    hasattr(part, "data") and part.mimetype and "audio" in part.mimetype
                ):
                    audio_data = part.data
                else:
                    # Not audio, pass through
                    yield part
                    continue

                logger.info(
                    f"🎤 WhisperProcessor: Processing {len(audio_data)} bytes of type {type(audio_data)}"
                )

                # WhisperLive expects base64 encoded audio
                # Ensure audio_data is bytes
                if isinstance(audio_data, (bytearray, memoryview)):
                    audio_data = bytes(audio_data)
                elif not isinstance(audio_data, bytes):
                    logger.error(f"Unexpected audio data type: {type(audio_data)}")
                    continue

                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                logger.debug(f"Sending {len(audio_base64)} base64 chars to WhisperLive")

                # Send to WhisperLive
                await self.whisper_ws.send(audio_base64)

                # Wait a bit for transcription with timeout
                try:
                    # Collect transcriptions for a short period
                    transcriptions = []
                    wait_time = 0.5  # Wait up to 500ms for transcriptions
                    start_time = time.time()

                    while time.time() - start_time < wait_time:
                        try:
                            trans_data = await asyncio.wait_for(
                                self.transcription_queue.get(), timeout=0.1
                            )
                            transcriptions.append(trans_data)
                        except asyncio.TimeoutError:
                            # No more transcriptions in queue
                            break

                    # Combine transcriptions
                    if transcriptions:
                        combined_text = " ".join([t["text"] for t in transcriptions])
                        languages = [
                            t.get("language")
                            for t in transcriptions
                            if t.get("language")
                        ]
                        detected_language = languages[0] if languages else "unknown"

                        logger.info(
                            f"🎤 WhisperLive transcription [{detected_language}]: '{combined_text}'"
                        )

                        # Yield the transcription
                        yield content_api.ProcessorPart(
                            text=combined_text,
                            mime_type="text/plain",
                            metadata={
                                "session_id": self.session_id,
                                "stage": "stt",
                                "confidence": 0.95,
                                "processor": "WhisperProcessor",
                                "language": detected_language,
                                "original_audio_size": len(audio_data),
                            },
                        )
                    else:
                        logger.warning("🎤 No transcription received from WhisperLive")

                        # Yield empty result with metadata
                        yield content_api.ProcessorPart(
                            text="",
                            mime_type="text/plain",
                            metadata={
                                "session_id": self.session_id,
                                "stage": "stt",
                                "confidence": 0.0,
                                "processor": "WhisperProcessor",
                                "error": "No transcription received",
                                "original_audio_size": len(audio_data),
                            },
                        )

                except Exception as e:
                    logger.error(f"Error processing transcription: {e}")

                    # Yield error result
                    yield content_api.ProcessorPart(
                        text="",
                        mime_type="text/plain",
                        metadata={
                            "session_id": self.session_id,
                            "stage": "stt",
                            "confidence": 0.0,
                            "processor": "WhisperProcessor",
                            "error": str(e),
                            "original_audio_size": len(audio_data),
                        },
                    )

        finally:
            await self._disconnect()


# Alternative implementation using the newer ProcessorPart API
class WhisperProcessorV2(processor.Processor):
    """Alternative implementation with better error handling"""

    def __init__(self, session_id: str, whisper_url: str = "ws://whisper-live:9090"):
        self.session_id = session_id
        self.whisper_url = whisper_url
        logger.info(f"WhisperProcessorV2 initialized for {session_id}")

    async def call(
        self, input_stream: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Simplified implementation focusing on reliability"""

        # Create a single WebSocket connection per call
        ws = None
        try:
            # Connect to WhisperLive
            ws_url = self.whisper_url.replace("http://", "ws://").replace(
                "https://", "wss://"
            )
            if not ws_url.endswith("/ws"):
                ws_url = f"{ws_url}/ws"

            ws = await websockets.connect(ws_url)

            # Send config
            await ws.send(
                json.dumps(
                    {
                        "uid": f"maestro_{self.session_id}_{int(time.time())}",
                        "language": None,
                        "task": "transcribe",
                        "model": "small",
                        "use_vad": True,
                    }
                )
            )

            async for part in input_stream:
                # Extract audio data
                audio_data = None
                if (
                    hasattr(part, "data")
                    and part.mime_type
                    and "audio" in part.mime_type
                ):
                    audio_data = part.data
                elif (
                    hasattr(part, "bytes")
                    and hasattr(part, "mimetype")
                    and "audio" in part.mimetype
                ):
                    audio_data = part.bytes

                if audio_data:
                    # Send audio
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                    await ws.send(audio_base64)

                    # Wait for response
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=5.0)

                        # Parse response
                        text = ""
                        language = "unknown"

                        if isinstance(response, str):
                            try:
                                data = json.loads(response)
                                if "text" in data:
                                    text = data["text"].strip()
                                    language = data.get("language", "unknown")
                                elif "segments" in data and data["segments"]:
                                    text = " ".join(
                                        seg.get("text", "") for seg in data["segments"]
                                    ).strip()
                                    language = data["segments"][0].get(
                                        "language", "unknown"
                                    )
                            except:
                                text = response.strip()

                        if text:
                            logger.info(f"🎤 Transcription: '{text}'")

                            # Create proper ProcessorPart
                            yield content_api.ProcessorPart(
                                data=text.encode("utf-8"),
                                mime_type="text/plain",
                                metadata={
                                    "text": text,  # Store text in metadata too
                                    "session_id": self.session_id,
                                    "stage": "stt",
                                    "processor": "WhisperProcessorV2",
                                    "language": language,
                                },
                            )
                        else:
                            # Empty transcription
                            yield content_api.ProcessorPart(
                                data=b"",
                                mime_type="text/plain",
                                metadata={
                                    "text": "",
                                    "session_id": self.session_id,
                                    "stage": "stt",
                                    "processor": "WhisperProcessorV2",
                                    "error": "Empty transcription",
                                },
                            )

                    except asyncio.TimeoutError:
                        logger.warning("WhisperLive timeout")
                        yield content_api.ProcessorPart(
                            data=b"",
                            mime_type="text/plain",
                            metadata={
                                "text": "",
                                "session_id": self.session_id,
                                "stage": "stt",
                                "processor": "WhisperProcessorV2",
                                "error": "Timeout",
                            },
                        )
                else:
                    # Pass through non-audio parts
                    yield part

        except Exception as e:
            logger.error(f"WhisperProcessorV2 error: {e}")
            # Yield error
            yield content_api.ProcessorPart(
                data=str(e).encode("utf-8"),
                mime_type="text/plain",
                metadata={
                    "session_id": self.session_id,
                    "stage": "stt",
                    "processor": "WhisperProcessorV2",
                    "error": str(e),
                },
            )
        finally:
            if ws:
                await ws.close()


# Usage in your VoiceSession
"""
Replace your WhisperProcessor initialization in VoiceSession with:

self.whisper = WhisperProcessorV2(session_id)

Or if you need the full async version:

self.whisper = WhisperProcessor(session_id)
"""

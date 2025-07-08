
import asyncio
import json
import logging
import time
import websockets
from config import config

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self, session_id: str, event_callback):
        self.session_id = session_id
        self.event_callback = event_callback
        self.websocket: websockets.WebSocketClientProtocol = None
        self.is_connected = False
        
        # Smart segment completion tracking
        self.last_segment_text = ""
        self.last_segment_time = 0
        self.segment_stability_duration = 1.5  # seconds to wait for segment stability - matches expectations
        self.pending_segment_timer = None
        
        # Prevent duplicate transcript processing
        self.last_sent_transcript = ""
        self.transcript_cooldown = 3.0  # seconds before allowing same transcript again
        
        # Prevent completed segment flood logging
        self.last_logged_completed_segment = ""
        self.completed_segment_log_count = 0
        
        # Prevent incomplete segment flood logging
        self.last_logged_incomplete_segment = ""
        self.incomplete_segment_log_count = 0

    async def connect(self):
        try:
            logger.info(f"STTService attempting to connect to {config.WHISPER_URL}")
            self.websocket = await asyncio.wait_for(websockets.connect(config.WHISPER_URL), timeout=10)
            logger.info(f"STTService WebSocket connected for session {self.session_id}")
            
            # WhisperLive specific configuration with optimized settings
            config_msg = {
                "uid": self.session_id,
                "language": "en",
                "task": "transcribe",
                "model": config.STT_MODEL,
                "use_vad": True,
                "no_speech_thresh": 0.15,  # Ultra-low threshold for immediate completion
                "send_last_n_segments": 1,  # Send immediately, don't accumulate
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
                
                # 🔧 CRITICAL FIX: Intelligent logging to prevent completed segment flood
                should_log = self._should_log_segment_data(data)
                if should_log:
                    logger.info(f"STTService received raw data for session {self.session_id}: {data}")
                
                # Check for segments and extract transcript
                if data.get("segments"):
                    segments = data.get("segments")
                    
                    # 🔧 FLOOD FIX: Only log segment count if we should log this data
                    if should_log:
                        logger.info(f"STTService found {len(segments)} segments for session {self.session_id}")
                    
                    # Process segments intelligently - don't rely on 'completed' flag
                    await self._process_segments_intelligently(segments)
                else:
                    if should_log:
                        logger.info(f"STTService no segments found in data for session {self.session_id}")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"STTService connection closed for session {self.session_id}.")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Error in STTService listener for session {self.session_id}: {e}")

    def _should_log_segment_data(self, data):
        """
        🔧 CRITICAL FIX: Prevent completed segment flood logging
        Only log segment data if it's new or meaningful to avoid hundreds of duplicate logs
        """
        # Check if this data contains completed segments
        segments = data.get("segments", [])
        if not segments:
            return True  # Always log non-segment data
            
        # Get the latest segment text and completion status
        latest_segment = segments[-1]
        segment_text = latest_segment.get('text', '').strip()
        is_completed = latest_segment.get('completed', False)
        
        # Handle both completed and incomplete segment flooding
        if is_completed:
            # For completed segments, check if we've already logged this exact text
            if segment_text == self.last_logged_completed_segment:
                # Only log every 50th duplicate to show it's still happening without flooding
                self.completed_segment_log_count += 1
                if self.completed_segment_log_count % 50 == 0:
                    logger.warning(f"STTService: WhisperLive sent the same completed segment {self.completed_segment_log_count} times: '{segment_text}'")
                return False  # Don't log the duplicate
            else:
                # New completed segment - reset counter and log it
                self.last_logged_completed_segment = segment_text
                self.completed_segment_log_count = 1
                return True
        else:
            # For incomplete segments, also prevent flooding
            if segment_text == self.last_logged_incomplete_segment:
                # Only log every 10th duplicate for incomplete segments (more frequent than completed)
                self.incomplete_segment_log_count += 1
                if self.incomplete_segment_log_count % 10 == 0:
                    logger.debug(f"STTService: Processing incomplete segment (#{self.incomplete_segment_log_count}): '{segment_text[:50]}...'")
                return False  # Don't log the duplicate
            else:
                # New incomplete segment - reset counter and log it
                self.last_logged_incomplete_segment = segment_text
                self.incomplete_segment_log_count = 1
                return True
            
        # For completed segments, check if we've already logged this exact text
        if segment_text == self.last_logged_completed_segment:
            # Only log every 50th duplicate to show it's still happening without flooding
            self.completed_segment_log_count += 1
            if self.completed_segment_log_count % 50 == 0:
                logger.warning(f"STTService: WhisperLive sent the same completed segment {self.completed_segment_log_count} times: '{segment_text}'")
            return False  # Don't log the duplicate
        else:
            # New completed segment - reset counter and log it
            self.last_logged_completed_segment = segment_text
            self.completed_segment_log_count = 1
            return True

    async def _process_segments_intelligently(self, segments):
        """
        Process segments without relying on 'completed' flag.
        Uses intelligent detection based on text stability and timing.
        """
        if not segments:
            return
            
        # Get the latest segment
        latest_segment = segments[-1]
        segment_text = latest_segment.get('text', '').strip()
        
        # Skip empty segments
        if not segment_text:
            logger.debug(f"STTService skipping empty segment for session {self.session_id}")
            return
            
        current_time = time.time()
        
        # CRITICAL FIX: If we already sent this exact transcript, don't process again
        if hasattr(self, 'last_sent_transcript') and segment_text == self.last_sent_transcript:
            logger.debug(f"STTService skipping already processed transcript: '{segment_text}'")
            return
        
        # Check if this is a genuinely new or significantly different segment
        if segment_text != self.last_segment_text:
            logger.info(f"STTService new segment text for session {self.session_id}: '{segment_text}'")
            self.last_segment_text = segment_text
            self.last_segment_time = current_time
            
            # Cancel any pending timer
            if self.pending_segment_timer:
                self.pending_segment_timer.cancel()
            
            # Set up new timer for segment completion
            self.pending_segment_timer = asyncio.create_task(
                self._complete_segment_after_delay(segment_text, self.segment_stability_duration)
            )
        else:
            # Same text, check if enough time has passed for natural completion
            time_since_last = current_time - self.last_segment_time
            
            # If segment text is stable for a reasonable period AND contains meaningful content, complete it
            if time_since_last >= self.segment_stability_duration and len(segment_text.split()) >= 2:
                logger.info(f"STTService segment stable for {time_since_last:.2f}s, completing: '{segment_text}'")
                await self._send_completed_transcript(segment_text)
                
                # Reset tracking
                self.last_segment_text = ""
                self.last_segment_time = 0
                
                # Cancel pending timer
                if self.pending_segment_timer:
                    self.pending_segment_timer.cancel()
                    self.pending_segment_timer = None

    async def _complete_segment_after_delay(self, segment_text: str, delay: float):
        """Complete a segment after a delay if no new updates arrive"""
        try:
            await asyncio.sleep(delay)
            
            # If we reach here, the segment hasn't been updated for 'delay' seconds
            if segment_text == self.last_segment_text and len(segment_text.split()) >= 2:
                logger.info(f"STTService auto-completing stable segment after {delay}s: '{segment_text}'")
                await self._send_completed_transcript(segment_text)
                
                # Reset tracking
                self.last_segment_text = ""
                self.last_segment_time = 0
                self.pending_segment_timer = None
                
        except asyncio.CancelledError:
            logger.debug(f"STTService segment completion timer cancelled for session {self.session_id}")

    async def _send_completed_transcript(self, transcript: str):
        """Send a completed transcript to the session manager"""
        if transcript and transcript.strip():
            transcript_clean = transcript.strip()
            
            # 🔧 CRITICAL FIX: Bulletproof duplicate prevention for transcript.final events
            if hasattr(self, 'last_sent_transcript') and transcript_clean == self.last_sent_transcript:
                logger.debug(f"STTService preventing duplicate transcript.final event: '{transcript_clean}'")
                return  # Don't send duplicate transcript.final events
            
            # Record that we sent this transcript
            self.last_sent_transcript = transcript_clean
            
            logger.info(f"STTService sending transcript.final event for session {self.session_id}: '{transcript_clean}'")
            # Fire an event back to the SessionManager
            await self.event_callback({
                "type": "transcript.final",
                "data": {"transcript": transcript_clean}
            })
        else:
            logger.warning(f"STTService attempted to send empty transcript for session {self.session_id}")

    async def send_audio(self, audio_chunk: bytes):
        if self.is_connected:
            await self.websocket.send(audio_chunk)

    async def close(self):
        if self.is_connected:
            self.is_connected = False
            
            # Cancel any pending segment timer
            if self.pending_segment_timer:
                self.pending_segment_timer.cancel()
                self.pending_segment_timer = None
            
            await self.websocket.close()
            logger.info(f"STTService connection for session {self.session_id} closed.")

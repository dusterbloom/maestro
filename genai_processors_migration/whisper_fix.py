class WhisperProcessor(processor.Processor):
    """Real WhisperLive processor using genai-processors framework"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.whisper_ws = None
        logger.info(f"WhisperProcessor initialized for {session_id}")
    
    async def _connect_whisper(self):
        """Connect to WhisperLive WebSocket"""
        try:
            if not self.whisper_ws:
                # Convert HTTP URL to WebSocket URL
                ws_url = config.WHISPER_URL.replace('http://', 'ws://').replace('https://', 'wss://')
                self.whisper_ws = await websockets.connect(f"{ws_url}/ws")
                logger.info("🔗 Connected to WhisperLive WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to WhisperLive: {e}")
            self.whisper_ws = None
    
    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """Send audio to WhisperLive and get transcription"""
        try:
            await self._connect_whisper()
            
            if not self.whisper_ws:
                raise Exception("WhisperLive connection failed")
            
            # Send audio data to WhisperLive
            await self.whisper_ws.send(audio_data)
            
            # Wait for transcription response with timeout
            response = await asyncio.wait_for(self.whisper_ws.recv(), timeout=10.0)
            
            # Parse WhisperLive response
            if isinstance(response, str):
                try:
                    data = json.loads(response)
                    transcript = data.get('text', '').strip()
                    if transcript:
                        logger.info(f"🎤 WhisperLive transcription: '{transcript}'")
                        return transcript
                except json.JSONDecodeError:
                    # Response might be plain text
                    transcript = response.strip()
                    if transcript:
                        logger.info(f"🎤 WhisperLive transcription (plain): '{transcript}'")
                        return transcript
            
            # If no valid transcript, return empty
            return ""
            
        except asyncio.TimeoutError:
            logger.warning("WhisperLive transcription timeout")
            return ""
        except Exception as e:
            logger.error(f"WhisperLive transcription error: {e}")
            # Close connection on error to force reconnect next time
            if self.whisper_ws:
                await self.whisper_ws.close()
                self.whisper_ws = None
            return ""
    
    async def call(self, input_stream: AsyncIterable[content_api.ProcessorPart]) -> AsyncIterable[content_api.ProcessorPart]:
        """Process audio through WhisperLive - real implementation"""
        async for part in input_stream:
            if content_api.is_audio(part.mimetype):
                logger.info(f"🎤 WhisperProcessor: Processing {len(part.bytes or b'')} bytes")
                
                # Get REAL transcription from WhisperLive
                transcript = await self._transcribe_audio(part.bytes or b'')
                
                if transcript:
                    yield content_api.ProcessorPart(
                        transcript,
                        mimetype="text/plain",
                        metadata={
                            "session_id": self.session_id,
                            "stage": "stt",
                            "confidence": 0.95,
                            "processor": "WhisperProcessor",
                            "original_audio_size": len(part.bytes or b''),
                            "transcript_length": len(transcript)
                        }
                    )
                else:
                    # If transcription failed or empty, yield empty result
                    logger.warning("🎤 WhisperProcessor: No transcription received")
                    yield content_api.ProcessorPart(
                        "",
                        mimetype="text/plain",
                        metadata={
                            "session_id": self.session_id,
                            "stage": "stt",
                            "confidence": 0.0,
                            "processor": "WhisperProcessor",
                            "error": "No transcription received",
                            "original_audio_size": len(part.bytes or b'')
                        }
                    )
            else:
                yield part

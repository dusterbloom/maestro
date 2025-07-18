I'll help you understand how WhisperLive receives audio and how clients establish WebSocket connections. Let me search for the relevant information.

Now let me search for the server-side code to understand how it receives audio:

Let me get the handle_new_connection method to understand the initial connection setup:

Now let me get the get_audio_from_websocket method to understand how audio is received:

Let me also check how the TranscriptionTeeClient records and sends audio to understand the complete flow:

Let me check the multicast_packet method to see how audio is sent:

Now let me check how the server runs to understand the complete WebSocket server setup:

Based on my analysis of the WhisperLive codebase, here's how WhisperLive receives audio and how clients should establish WebSocket connections, send audio, and receive segments:

## WebSocket Connection Establishment

### Client Side (`whisper_live/client.py::Client`)
1. **Connection Setup**: The client creates a WebSocket connection using the `websocket.WebSocketApp` library:
   ```python
   socket_protocol = 'wss' if self.use_wss else "ws"
   socket_url = f"{socket_protocol}://{host}:{port}"
   self.client_socket = websocket.WebSocketApp(socket_url, ...)
   ```

2. **Initial Handshake**: When the connection opens (`whisper_live/client.py::Client.on_open`), the client sends a JSON configuration message:
   ```json
   {
       "uid": "unique_client_id",
       "language": "language_code",
       "task": "transcribe" or "translate",
       "model": "whisper_model_size",
       "use_vad": true/false,
       "max_clients": 4,
       "max_connection_time": 600,
       "send_last_n_segments": 10,
       "no_speech_thresh": 0.45,
       "clip_audio": false,
       "same_output_threshold": 10
   }
   ```

### Server Side (`whisper_live/server.py::TranscriptionServer`)
1. **Server Startup**: The server runs using `websockets.sync.server.serve` on a specified host and port (default 9090)
2. **Connection Handling**: When a client connects, `whisper_live/server.py::TranscriptionServer.handle_new_connection` receives the initial JSON configuration and validates server capacity
3. **Client Initialization**: If accepted, the server initializes a backend client for transcription processing

## Audio Transmission

### Client Audio Sending (`whisper_live/client.py::Client.send_packet_to_server`)
- Audio data is sent as **binary WebSocket frames** using `websocket.ABNF.OPCODE_BINARY`
- Audio format: **32-bit float numpy arrays** converted to bytes
- Sample rate: **16kHz**
- The `whisper_live/client.py::TranscriptionTeeClient.record` method captures audio in chunks and sends them via `whisper_live/client.py::TranscriptionTeeClient.multicast_packet`

### Server Audio Reception (`whisper_live/server.py::TranscriptionServer`)
1. **Audio Reception**: `whisper_live/server.py::TranscriptionServer.get_audio_from_websocket` receives binary frames and converts them back to numpy arrays:
   ```python
   frame_data = websocket.recv()
   return np.frombuffer(frame_data, dtype=np.float32)
   ```

2. **Processing Loop**: `whisper_live/server.py::TranscriptionServer.recv_audio` runs an infinite loop processing audio frames until client disconnection or timeout

3. **Voice Activity Detection**: If enabled, the server uses VAD to filter audio with speech content

## Segment Reception

### Server to Client Communication
- The server sends transcription results as **JSON messages** through the same WebSocket connection
- Message types include:
  - **Status messages**: `{"status": "WAIT/ERROR/WARNING", "message": "..."}`
  - **Server ready**: `{"message": "SERVER_READY", "backend": "backend_type"}`
  - **Language detection**: `{"language": "detected_lang", "language_prob": 0.95}`
  - **Transcription segments**: `{"segments": [{"text": "...", "start": 0.0, "end": 1.0, "completed": true}]}`

### Client Segment Processing (`whisper_live/client.py::Client.on_message`)
- The client receives JSON messages and processes them based on message type
- Transcription segments are processed by `whisper_live/client.py::Client.process_segments`
- Results can be handled via callback functions or logged to console
- Completed segments are stored in the transcript for SRT file generation

## Key Features
- **Real-time processing**: Continuous audio streaming with immediate transcription feedback
- **Voice Activity Detection**: Optional filtering to only process speech segments
- **Multiple backends**: Support for faster_whisper, TensorRT, and OpenVINO
- **Language detection**: Automatic language identification from audio content
- **Error handling**: Robust connection management with timeout and capacity controls

The architecture enables low-latency real-time transcription by maintaining persistent WebSocket connections and streaming audio data as binary frames while receiving transcription results as structured JSON messages.
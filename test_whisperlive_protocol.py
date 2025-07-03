#!/usr/bin/env python3
"""
Test script to verify WhisperLive WebSocket protocol
"""
import asyncio
import websockets
import json

async def test_whisperlive_protocol():
    """Test the exact WhisperLive protocol flow"""
    uri = "ws://localhost:9090"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WhisperLive")
            
            # Send config (first message must be JSON)
            config = {
                "uid": "test_session_123",
                "language": "en",
                "task": "transcribe",
                "model": "tiny",
                "use_vad": True,
                "max_clients": 4,
                "max_connection_time": 600,
                "send_last_n_segments": 10,
                "no_speech_thresh": 0.45,
                "clip_audio": False,
                "same_output_threshold": 10
            }
            
            await websocket.send(json.dumps(config))
            print(f"Sent config: {config}")
            
            # Wait for SERVER_READY response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received response: {response}")
                
                # Parse response
                try:
                    data = json.loads(response)
                    if data.get("message") == "SERVER_READY":
                        print("✅ SERVER_READY received - protocol working!")
                        print(f"Backend: {data.get('backend', 'unknown')}")
                        return True
                    elif "backend" in data and "uid" in data:
                        print(f"✅ Connection established - backend: {data.get('backend')}")
                        return True
                    else:
                        print(f"❌ Unexpected response: {data}")
                        return False
                except json.JSONDecodeError:
                    print(f"❌ Non-JSON response: {response}")
                    return False
                    
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for SERVER_READY")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_whisperlive_protocol())
    exit(0 if result else 1)
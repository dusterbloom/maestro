#!/usr/bin/env python3
import asyncio
import websockets
import json

async def test_connection():
    try:
        # Use environment variable for WebSocket URL
        ws_url = os.getenv("WHISPER_WS_URL", "ws://localhost:8000/ws")
        print(f"Attempting to connect to {ws_url}")
        
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            print("✅ Connected successfully")
            
            # Send config like the frontend
            config = {
                "uid": f"debug_test_{int(asyncio.get_event_loop().time() * 1000)}",
                "language": "en",
                "task": "transcribe", 
                "model": "tiny",
                "use_vad": True,
                "multilingual": False
            }
            
            await websocket.send(json.dumps(config))
            print(f"📤 Sent config: {config}")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                print(f"📥 Received: {response}")
                
                # Keep connection alive briefly
                await asyncio.sleep(2)
                print("✅ Connection maintained successfully")
                
            except asyncio.TimeoutError:
                print("⚠️ No response received within 5 seconds")
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e.code} {e.reason}")
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ Invalid status code: {e.status_code}")
    except ConnectionRefusedError:
        print("❌ Connection refused - is the orchestrator running?")
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
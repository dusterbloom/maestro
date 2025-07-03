#!/usr/bin/env python3
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Send config like WhisperLive
            config = {
                "uid": "test_session_123",
                "language": "en",
                "task": "transcribe",
                "model": "tiny",
                "use_vad": True,
                "multilingual": False
            }
            await websocket.send(json.dumps(config))
            print(f"Sent config: {config}")
            
            # Wait for response
            response = await websocket.recv()
            print(f"Received: {response}")
            
            # Keep connection alive for a few seconds
            await asyncio.sleep(2)
            print("Connection maintained successfully")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
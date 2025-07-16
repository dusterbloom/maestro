try:
    import asyncio
    import websockets

    async def test_websocket():
        uri = "ws://localhost:8000/ws"
        print(f"Attempting to connect to {uri}")
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to WebSocket")
                greeting = await websocket.recv()
                print(f"Received: {greeting}")
        except Exception as e:
            print(f"Error: {e}")
        print("Test completed")

    asyncio.get_event_loop().run_until_complete(test_websocket())
    print("Script execution finished")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

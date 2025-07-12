#!/usr/bin/env python3
"""
Test script to debug ProcessorPart issues
"""

import sys
import traceback

try:
    from genai_processors import content_api, processor
    print("✅ Successfully imported genai_processors")
    
    # Test creating a simple ProcessorPart
    print("🧪 Testing ProcessorPart creation...")
    
    # Test 1: Simple text ProcessorPart
    try:
        text_part = content_api.ProcessorPart("Hello World")
        print(f"✅ Simple text part: {text_part}")
        print(f"   text: {text_part.text}")
        print(f"   mimetype: {text_part.mimetype}")
        print(f"   _part type: {type(text_part._part)}")
    except Exception as e:
        print(f"❌ Simple text part failed: {e}")
        traceback.print_exc()
    
    # Test 2: ProcessorPart with keyword arguments  
    try:
        text_part_with_meta = content_api.ProcessorPart(
            "Hello GenAI",
            mimetype="text/plain",
            metadata={"test": "value"}
        )
        print(f"✅ Text part with metadata: {text_part_with_meta}")
    except Exception as e:
        print(f"❌ Text part with metadata failed: {e}")
        traceback.print_exc()
    
    # Test 3: ProcessorPart with bytes (audio simulation)
    try:
        audio_bytes = b"fake_audio_data"
        audio_part = content_api.ProcessorPart(
            audio_bytes,
            mimetype="audio/wav",
            metadata={"source": "test"}
        )
        print(f"✅ Audio part: {audio_part}")
        print(f"   bytes length: {len(audio_part.bytes or b'')}")
    except Exception as e:
        print(f"❌ Audio part failed: {e}")
        traceback.print_exc()
        
    # Test 4: Test async iteration
    try:
        async def test_async():
            async def simple_stream():
                yield text_part
                yield audio_part
                
            parts = []
            async for part in simple_stream():
                parts.append(part)
                print(f"   Received part: {part.mimetype}")
            print(f"✅ Async iteration worked, got {len(parts)} parts")
            
        import asyncio
        asyncio.run(test_async())
    except Exception as e:
        print(f"❌ Async iteration failed: {e}")
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ Failed to import genai_processors: {e}")
    print("Make sure genai-processors is installed: pip install genai-processors")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("🎉 All tests completed!")

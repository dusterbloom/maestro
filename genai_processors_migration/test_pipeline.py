#!/usr/bin/env python3
"""
Test script to simulate the full pipeline and find the ProcessorPart error
"""

import asyncio
import sys
import traceback

try:
    from genai_processors import content_api, processor
    print("✅ Successfully imported genai_processors")
    
    # Create mock processors similar to the real ones
    class MockWhisperProcessor(processor.Processor):
        def __init__(self, session_id: str):
            self.session_id = session_id
            print(f"MockWhisperProcessor initialized for {session_id}")
        
        async def call(self, input_stream):
            async for part in input_stream:
                print(f"WhisperProcessor received: {type(part)} with mimetype {getattr(part, 'mimetype', 'N/A')}")
                if content_api.is_audio(part.mimetype):
                    transcript = f"Mock transcript for {len(part.bytes or b'')} bytes"
                    yield content_api.ProcessorPart(
                        transcript,
                        mimetype="text/plain",
                        metadata={
                            "session_id": self.session_id,
                            "stage": "stt",
                            "processor": "MockWhisperProcessor"
                        }
                    )
                else:
                    yield part

    class MockOllamaProcessor(processor.Processor):
        def __init__(self, session_id: str):
            self.session_id = session_id
            print(f"MockOllamaProcessor initialized for {session_id}")
        
        async def call(self, input_stream):
            async for part in input_stream:
                print(f"OllamaProcessor received: {type(part)} with mimetype {getattr(part, 'mimetype', 'N/A')}")
                if content_api.is_text(part.mimetype):
                    text = content_api.as_text(part)
                    llm_response = f"Mock LLM response to: {text[:50]}..."
                    yield content_api.ProcessorPart(
                        llm_response,
                        mimetype="text/plain",
                        metadata={
                            "session_id": self.session_id,
                            "stage": "llm",
                            "processor": "MockOllamaProcessor"
                        }
                    )
                else:
                    yield part

    class MockTTSProcessor(processor.Processor):
        def __init__(self, session_id: str):
            self.session_id = session_id
            print(f"MockTTSProcessor initialized for {session_id}")
        
        async def call(self, input_stream):
            async for part in input_stream:
                print(f"TTSProcessor received: {type(part)} with mimetype {getattr(part, 'mimetype', 'N/A')}")
                if content_api.is_text(part.mimetype):
                    text = content_api.as_text(part)
                    # Simulate audio generation
                    fake_audio = b"fake_audio_data_for_" + text[:20].encode()
                    yield content_api.ProcessorPart(
                        fake_audio,
                        mimetype="audio/wav",
                        metadata={
                            "session_id": self.session_id,
                            "stage": "tts",
                            "text": text,
                            "processor": "MockTTSProcessor"
                        }
                    )
                else:
                    yield part

    async def test_full_pipeline():
        print("\n🚀 Testing full pipeline...")
        
        session_id = "test_session_123"
        
        # Create processors
        whisper = MockWhisperProcessor(session_id)
        ollama = MockOllamaProcessor(session_id)
        tts = MockTTSProcessor(session_id)
        
        # Create input audio data
        audio_data = b"fake_audio_input_data"
        
        # Create ProcessorPart using REAL genai-processors API
        audio_part = content_api.ProcessorPart(
            audio_data,
            mimetype="audio/wav",
            metadata={"session_id": session_id, "stage": "input"}
        )
        print(f"✅ Created audio_part: {audio_part}")
        
        # Create async stream using REAL genai-processors API
        async def create_input_stream():
            yield audio_part
        
        input_stream = create_input_stream()
        print("✅ Created input_stream")
        
        # Process through the pipeline
        print("\n🔄 Step 1: WhisperProcessor")
        whisper_stream = whisper.call(input_stream)
        
        print("🔄 Step 2: OllamaProcessor")
        ollama_stream = ollama.call(whisper_stream)
        
        print("🔄 Step 3: TTSProcessor")
        tts_stream = tts.call(ollama_stream)
        
        # Process final results
        print("🔄 Processing final results...")
        sequence = 1
        async for result in tts_stream:
            stage = result.metadata.get("stage", "unknown")
            processor_name = result.metadata.get("processor", "unknown")
            
            print(f"✨ Result from {processor_name}: {stage}, {result.mimetype}")
            print(f"   Result type: {type(result)}")
            print(f"   Has _part: {hasattr(result, '_part')}")
            if hasattr(result, '_part'):
                print(f"   _part type: {type(result._part)}")
            
            sequence += 1
        
        print("✅ Pipeline completed successfully!")

    # Run the test
    asyncio.run(test_full_pipeline())
    print("\n🎉 All pipeline tests completed!")
        
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1)

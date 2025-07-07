#!/usr/bin/env python3
"""
Test script to debug resemblyzer audio processing specifically
"""

import numpy as np
import tempfile
import os
from pathlib import Path

def test_resemblyzer_processing():
    print("=== Resemblyzer Audio Processing Test ===\n")
    
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        import soundfile as sf
        
        # Create encoder
        encoder = VoiceEncoder()
        print("✓ VoiceEncoder created")
        
        # Test with different audio durations and formats
        sample_rate = 16000
        durations = [0.1, 0.5, 1.0, 2.0]  # seconds
        
        for duration in durations:
            print(f"\n--- Testing {duration}s audio ---")
            
            # Create test audio - sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            print(f"Created audio: shape={audio_data.shape}, duration={duration}s, dtype={audio_data.dtype}")
            
            # Save as WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                
                # Test resemblyzer preprocessing
                try:
                    processed_audio = preprocess_wav(tmp_file.name)
                    print(f"✓ Preprocessed audio: shape={processed_audio.shape}")
                    
                    if len(processed_audio) > 0:
                        # Try to generate embedding
                        try:
                            embedding = encoder.embed_utterance(processed_audio)
                            print(f"✓ Generated embedding: shape={embedding.shape}")
                        except Exception as e:
                            print(f"✗ Embedding failed: {e}")
                    else:
                        print("⚠️ Preprocessed audio is empty!")
                        
                        # Debug: check the WAV file manually
                        try:
                            data, sr = sf.read(tmp_file.name)
                            print(f"   WAV file check: shape={data.shape}, sr={sr}, dtype={data.dtype}")
                            print(f"   Audio stats: min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}")
                        except Exception as e:
                            print(f"   WAV file read failed: {e}")
                
                except Exception as e:
                    print(f"✗ Preprocessing failed: {e}")
                
                # Clean up
                os.unlink(tmp_file.name)
        
        # Test with actual voice-like audio (more complex waveform)
        print(f"\n--- Testing voice-like audio ---")
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a more complex waveform that resembles speech
        fundamental = 150  # Hz - typical male voice
        audio_data = (
            np.sin(2 * np.pi * fundamental * t) * 0.5 +
            np.sin(2 * np.pi * fundamental * 2 * t) * 0.3 +
            np.sin(2 * np.pi * fundamental * 3 * t) * 0.2 +
            np.random.normal(0, 0.05, len(t))  # Add some noise
        ).astype(np.float32)
        
        # Add some amplitude modulation to simulate speech patterns
        envelope = np.abs(np.sin(2 * np.pi * 5 * t))  # 5 Hz modulation
        audio_data *= envelope
        
        print(f"Created voice-like audio: shape={audio_data.shape}")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            try:
                processed_audio = preprocess_wav(tmp_file.name)
                print(f"✓ Voice-like preprocessed: shape={processed_audio.shape}")
                
                if len(processed_audio) > 0:
                    embedding = encoder.embed_utterance(processed_audio)
                    print(f"✓ Voice-like embedding: shape={embedding.shape}")
                else:
                    print("⚠️ Voice-like audio also empty after preprocessing!")
            except Exception as e:
                print(f"✗ Voice-like processing failed: {e}")
            
            os.unlink(tmp_file.name)
    
    except Exception as e:
        print(f"✗ Test setup failed: {e}")

def test_audio_from_bytes():
    """Test processing audio from raw bytes (like in your service)"""
    print(f"\n=== Testing Audio from Bytes ===\n")
    
    try:
        import io
        import wave
        import soundfile as sf
        from resemblyzer import preprocess_wav, VoiceEncoder
        
        encoder = VoiceEncoder()
        
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        # Convert to bytes (like your service receives)
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        
        print(f"Created audio bytes: {len(audio_bytes)} bytes")
        
        # Method 1: Save bytes to temp file and process (like your current approach)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            
            try:
                processed = preprocess_wav(tmp_file.name)
                print(f"✓ Method 1 (temp file): shape={processed.shape}")
                
                if len(processed) > 0:
                    embedding = encoder.embed_utterance(processed)
                    print(f"✓ Method 1 embedding: shape={embedding.shape}")
            except Exception as e:
                print(f"✗ Method 1 failed: {e}")
            
            os.unlink(tmp_file.name)
        
        # Method 2: Process directly from bytes
        try:
            buffer = io.BytesIO(audio_bytes)
            data, sr = sf.read(buffer)
            print(f"✓ Method 2 (direct): shape={data.shape}, sr={sr}")
            
            # Ensure mono and correct format for resemblyzer
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resemblyzer expects specific preprocessing
            if sr != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            
            # Generate embedding directly
            embedding = encoder.embed_utterance(data)
            print(f"✓ Method 2 embedding: shape={embedding.shape}")
            
        except Exception as e:
            print(f"✗ Method 2 failed: {e}")
    
    except Exception as e:
        print(f"✗ Bytes test setup failed: {e}")

if __name__ == "__main__":
    test_resemblyzer_processing()
    test_audio_from_bytes()
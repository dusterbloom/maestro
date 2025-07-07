#!/usr/bin/env python3
"""
Debug script to test Resemblyzer embedding generation
Run this inside the Docker container to isolate the issue
"""
import io
import wave
import numpy as np
import tempfile
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import soundfile as sf
    from resemblyzer import VoiceEncoder, preprocess_wav
    logger.info("✅ Successfully imported soundfile and resemblyzer")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    exit(1)

def create_test_wav_data(duration_seconds=2.0, sample_rate=16000, frequency=440):
    """Create test WAV data with a sine wave"""
    logger.info(f"🎵 Generating {duration_seconds}s test audio at {sample_rate}Hz")
    
    # Generate sine wave
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    sine_wave = 0.3 * np.sin(2 * np.pi * frequency * t)  # 440Hz tone at 30% volume
    
    # Convert to int16 PCM
    int16_data = (sine_wave * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_data.tobytes())
    
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.read()
    
    logger.info(f"📊 Generated WAV: {len(wav_bytes)} bytes")
    return wav_bytes

def test_wav_reading(wav_bytes):
    """Test reading WAV data with soundfile"""
    logger.info("🔍 Testing WAV reading with soundfile...")
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(wav_bytes)
        temp_path = temp_file.name
    
    try:
        # Try to read with soundfile
        data, sr = sf.read(temp_path, dtype='float32')
        logger.info(f"✅ soundfile read successful: shape={data.shape}, sr={sr}, dtype={data.dtype}")
        
        # Test various operations
        if data.ndim > 1:
            logger.info(f"🔄 Multi-channel detected, converting to mono")
            data = data.mean(axis=1)
        
        duration = len(data) / sr
        rms = np.sqrt(np.mean(data ** 2))
        db = 20 * np.log10(rms + 1e-8)
        
        logger.info(f"📊 Audio stats: duration={duration:.2f}s, RMS={rms:.4f}, dB={db:.2f}")
        
        return data, sr
        
    except Exception as e:
        logger.error(f"❌ soundfile read failed: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        return None, None
    finally:
        Path(temp_path).unlink()

def test_resemblyzer_preprocessing(audio_data, sample_rate):
    """Test Resemblyzer preprocessing"""
    logger.info("🔧 Testing Resemblyzer preprocessing...")
    
    try:
        preprocessed = preprocess_wav(audio_data, sample_rate)
        logger.info(f"✅ Resemblyzer preprocess successful: {len(preprocessed)} samples")
        return preprocessed
    except Exception as e:
        logger.error(f"❌ Resemblyzer preprocess failed: {e}")
        return None

def test_resemblyzer_embedding(preprocessed_wav):
    """Test Resemblyzer embedding generation"""
    logger.info("🧠 Testing Resemblyzer embedding generation...")
    
    try:
        # Initialize encoder
        encoder = VoiceEncoder(device="cpu")  # Force CPU for compatibility
        logger.info("✅ VoiceEncoder initialized")
        
        # Generate embedding
        embedding = encoder.embed_utterance(preprocessed_wav)
        logger.info(f"✅ Embedding generated: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
        
        return embedding.tolist()
        
    except Exception as e:
        logger.error(f"❌ Resemblyzer embedding failed: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return None

def test_realistic_audio_buffer():
    """Test with audio buffer similar to your AudioBufferManager"""
    logger.info("🎤 Testing with realistic audio buffer simulation...")
    
    # Simulate 10 seconds of accumulated float32 audio chunks
    sample_rate = 16000
    duration = 10.0
    
    # Create float32 audio data (similar to what you get from browser)
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Mix of frequencies to simulate speech
    audio_float32 = (
        0.1 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
        0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
        0.1 * np.sin(2 * np.pi * 1600 * t)   # High frequency
    )
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, len(audio_float32))
    audio_float32 += noise
    
    logger.info(f"📊 Generated realistic audio: {len(audio_float32)} samples, dtype={audio_float32.dtype}")
    
    # Convert to WAV (similar to your get_buffer_as_wav method)
    int16_array = (audio_float32 * 32767).astype(np.int16)
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_array.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

def main():
    """Run all debug tests"""
    logger.info("🚀 Starting Resemblyzer debug tests...")
    
    # Test 1: Simple sine wave
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Simple sine wave")
    logger.info("="*50)
    
    wav_bytes = create_test_wav_data()
    audio_data, sr = test_wav_reading(wav_bytes)
    
    if audio_data is not None:
        preprocessed = test_resemblyzer_preprocessing(audio_data, sr)
        if preprocessed is not None:
            embedding = test_resemblyzer_embedding(preprocessed)
            if embedding:
                logger.info("✅ Test 1 PASSED: Simple sine wave embedding successful")
            else:
                logger.error("❌ Test 1 FAILED: Embedding generation failed")
        else:
            logger.error("❌ Test 1 FAILED: Preprocessing failed")
    else:
        logger.error("❌ Test 1 FAILED: WAV reading failed")
    
    # Test 2: Realistic audio buffer
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Realistic audio buffer")
    logger.info("="*50)
    
    realistic_wav = test_realistic_audio_buffer()
    audio_data2, sr2 = test_wav_reading(realistic_wav)
    
    if audio_data2 is not None:
        preprocessed2 = test_resemblyzer_preprocessing(audio_data2, sr2)
        if preprocessed2 is not None:
            embedding2 = test_resemblyzer_embedding(preprocessed2)
            if embedding2:
                logger.info("✅ Test 2 PASSED: Realistic audio embedding successful")
            else:
                logger.error("❌ Test 2 FAILED: Embedding generation failed")
        else:
            logger.error("❌ Test 2 FAILED: Preprocessing failed")
    else:
        logger.error("❌ Test 2 FAILED: WAV reading failed")
    
    logger.info("\n🏁 Debug tests completed!")

if __name__ == "__main__":
    main()

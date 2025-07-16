#!/usr/bin/env python3
"""
Updated server startup for GenAI Processors migration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Check required environment variables."""
    required_vars = {
        "GOOGLE_API_KEY": "Google API key for Gemini models",
        "GOOGLE_PROJECT_ID": "Google Cloud Project ID for STT/TTS"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            missing_vars.append(f"  {var}: {description}")
    
    if missing_vars:
        logger.warning("Missing environment variables:")
        for var in missing_vars:
            print(var)
        print("\nOptional: Set these in your .env file or environment")
    
    return len(missing_vars) == 0


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "genai_processors",
        "google.genai", 
        "fastapi",
        "uvicorn",
        "websockets",
        "pyaudio",
        "numpy",
        "aiohttp"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("Missing required packages:")
        for package in missing_packages:
            print(f"  {package}")
        print(f"\nInstall with: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Start the GenAI Processors server."""
    print("🚀 Starting GenAI Processors Gateway...")
    print("📦 Framework: google-gemini/genai-processors")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        print("⚠️  Some features may not work without proper environment setup")
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Start the server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "processors.gateway.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ]
    
    print("\n📡 Server starting on http://localhost:8000")
    print("🎤 WebSocket endpoints:")
    print("   - ws://localhost:8000/ws/{pipeline_name}")
    print("   - ws://localhost:8000/ws/live/{model_name}")
    
    print("\n📋 Available pipelines:")
    print("   Built-in Registry:")
    print("     - audio_to_text: Audio → Text conversion")
    print("     - text_to_audio: Text → Audio conversion") 
    print("     - full_voice: Complete voice pipeline")
    print("     - realtime_agent: Real-time Gemini agent")
    print("   Configuration-based:")
    print("     - voice_chat: STT + LLM processing")
    print("     - hybrid_agent: Fallback routing")
    print("     - local_voice_agent: Local-only processing")
    
    print("\n🎯 API Endpoints:")
    print("   - GET  /health - Health check")
    print("   - GET  /processors - List processors")
    print("   - GET  /pipelines - List pipelines") 
    print("   - GET  /pipelines/registry - Built-in templates")
    print("   - POST /pipelines/validate - Validate config")
    
    print("\n🔧 Testing:")
    print("   1. Check health: curl http://localhost:8000/health")
    print("   2. List pipelines: curl http://localhost:8000/pipelines")
    print("   3. WebSocket test: Connect to ws://localhost:8000/ws/audio_to_text")
    
    print("\n📚 Examples:")
    print("   - Basic STT: ws://localhost:8000/ws/audio_to_text")
    print("   - Voice chat: ws://localhost:8000/ws/full_voice")
    print("   - Live agent: ws://localhost:8000/ws/live/gemini-2.5-flash-preview-native-audio-dialog")
    
    print(f"\n🌟 Using genai-processors framework")
    print("   - Async streaming architecture")
    print("   - Processor chaining with + operator")
    print("   - Built-in Gemini API integration")
    print("   - Modular and extensible design")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if all dependencies are installed")
        print("   2. Verify environment variables are set")
        print("   3. Ensure ports 8000 is available")
        print("   4. Check logs for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
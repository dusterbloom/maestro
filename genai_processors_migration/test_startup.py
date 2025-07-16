#!/usr/bin/env python3
"""
Configuration diagnostics script to identify and fix pipeline issues.
"""

import os
import yaml
from pathlib import Path
import json

def check_config_files():
    """Check if configuration files exist and are valid."""
    config_dir = Path("config")
    
    print("🔍 Configuration File Diagnostics")
    print("=" * 50)
    
    # Check if config directory exists
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        print("💡 Creating config directory...")
        config_dir.mkdir(exist_ok=True)
        return False
    
    # Check processors.yaml
    processors_file = config_dir / "processors.yaml"
    if processors_file.exists():
        try:
            with open(processors_file, 'r') as f:
                processors_config = yaml.safe_load(f)
            print(f"✅ processors.yaml loaded successfully")
            print(f"   Found {len(processors_config.get('processors', {}))} processors")
        except Exception as e:
            print(f"❌ Error loading processors.yaml: {e}")
            return False
    else:
        print(f"❌ processors.yaml not found")
        return False
    
    # Check pipelines.yaml
    pipelines_file = config_dir / "pipelines.yaml"
    if pipelines_file.exists():
        try:
            with open(pipelines_file, 'r') as f:
                pipelines_config = yaml.safe_load(f)
            print(f"✅ pipelines.yaml loaded successfully")
            print(f"   Found {len(pipelines_config.get('pipelines', {}))} pipelines")
            
            # List all pipelines
            pipelines = pipelines_config.get('pipelines', {})
            print(f"\n📋 Available Pipelines:")
            for name, config in pipelines.items():
                print(f"   - {name}: {' → '.join(config)}")
                
        except Exception as e:
            print(f"❌ Error loading pipelines.yaml: {e}")
            return False
    else:
        print(f"❌ pipelines.yaml not found")
        return False
    
    return True

def validate_pipeline_references():
    """Validate that all pipeline processors exist."""
    print(f"\n🔗 Pipeline Reference Validation")
    print("=" * 50)
    
    config_dir = Path("config")
    
    try:
        # Load configurations
        with open(config_dir / "processors.yaml", 'r') as f:
            processors_config = yaml.safe_load(f)
        with open(config_dir / "pipelines.yaml", 'r') as f:
            pipelines_config = yaml.safe_load(f)
        
        processors = processors_config.get('processors', {})
        pipelines = pipelines_config.get('pipelines', {})
        
        all_valid = True
        
        for pipeline_name, processor_list in pipelines.items():
            print(f"\n📋 Pipeline: {pipeline_name}")
            for processor_name in processor_list:
                if processor_name in processors:
                    print(f"   ✅ {processor_name}")
                else:
                    print(f"   ❌ {processor_name} (NOT FOUND)")
                    all_valid = False
        
        if all_valid:
            print(f"\n✅ All pipeline references are valid!")
        else:
            print(f"\n❌ Some pipeline references are invalid!")
            
        return all_valid
        
    except Exception as e:
        print(f"❌ Error validating references: {e}")
        return False

def check_environment_variables():
    """Check required environment variables."""
    print(f"\n🌍 Environment Variables")
    print("=" * 50)
    
    required_vars = {
        "GOOGLE_API_KEY": "Google API key for Gemini models",
        "GOOGLE_PROJECT_ID": "Google Cloud Project ID for STT/TTS"
    }
    
    optional_vars = {
        "GEMINI_MODEL": "Gemini model name",
        "OLLAMA_MODEL": "Ollama model name", 
        "TTS_VOICE": "TTS voice selection",
        "KOKORO_URL": "Kokoro TTS service URL"
    }
    
    print("Required:")
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"   ✅ {var}: {description}")
        else:
            print(f"   ❌ {var}: {description} (NOT SET)")
    
    print("\nOptional:")
    for var, description in optional_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ⚪ {var}: {description} (not set)")

def test_imports():
    """Test if required modules can be imported."""
    print(f"\n📦 Import Testing")
    print("=" * 50)
    
    required_imports = [
        "genai_processors",
        "genai_processors.content_api",
        "genai_processors.processor", 
        "genai_processors.streams",
        "fastapi",
        "uvicorn",
        "yaml",
        "numpy"
    ]
    
    optional_imports = [
        "google.genai",
        "genai_processors.core.speech_to_text",
        "genai_processors.core.text_to_speech",
        "genai_processors.core.genai_model",
        "genai_processors.core.audio_io",
        "pyaudio",
        "aiohttp"
    ]
    
    print("Required imports:")
    for module in required_imports:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
    
    print("\nOptional imports:")
    for module in optional_imports:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ⚪ {module}: not available")

def create_minimal_config():
    """Create minimal working configuration files."""
    print(f"\n🔧 Creating Minimal Configuration")
    print("=" * 50)
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Minimal processors.yaml
    minimal_processors = {
        "processors": {
            "audio_preprocessor": {
                "module": "processors.audio.audio_processor",
                "class": "AudioProcessor",
                "params": {
                    "target_sample_rate": 16000,
                    "target_channels": 1
                }
            },
            "stt_router": {
                "module": "processors.stt.speech_to_text", 
                "class": "STTRouterProcessor",
                "params": {
                    "prefer_cloud": True
                }
            },
            "llm_router": {
                "module": "processors.llm.genai_model",
                "class": "LLMRouterProcessor", 
                "params": {
                    "prefer_gemini": True,
                    "system_instruction": "You are a helpful assistant."
                }
            },
            "tts_router": {
                "module": "processors.tts.text_to_speech",
                "class": "TTSRouterProcessor",
                "params": {
                    "prefer_google": True
                }
            }
        }
    }
    
    # Minimal pipelines.yaml
    minimal_pipelines = {
        "pipelines": {
            "stt_only": ["audio_preprocessor", "stt_router"],
            "stt_llm": ["audio_preprocessor", "stt_router", "llm_router"],
            "full_pipeline": ["audio_preprocessor", "stt_router", "llm_router", "tts_router"],
            "test_stt": ["audio_preprocessor", "stt_router"]
        }
    }
    
    # Write files
    with open(config_dir / "processors.yaml", 'w') as f:
        yaml.dump(minimal_processors, f, default_flow_style=False, indent=2)
    
    with open(config_dir / "pipelines.yaml", 'w') as f:
        yaml.dump(minimal_pipelines, f, default_flow_style=False, indent=2)
    
    print("✅ Created minimal configuration files")
    print("   - config/processors.yaml")
    print("   - config/pipelines.yaml")

def main():
    """Run all diagnostics."""
    print("🏥 GenAI Processors Configuration Diagnostics")
    print("=" * 60)
    
    # Run diagnostics
    config_ok = check_config_files()
    
    if config_ok:
        validate_pipeline_references()
    else:
        print("\n🔧 Configuration files missing or invalid.")
        create_minimal_config()
        print("\n✅ Minimal configuration created. Please set environment variables:")
        print("   export GOOGLE_API_KEY=your_api_key")
        print("   export GOOGLE_PROJECT_ID=your_project_id")
    
    check_environment_variables()
    test_imports()
    
    print(f"\n🎯 Summary & Next Steps:")
    print("1. Ensure all configuration files are present")
    print("2. Set required environment variables") 
    print("3. Install missing dependencies: pip install -r requirements.txt")
    print("4. Test with: python run_server.py")
    print("5. Check WebSocket: ws://localhost:8000/ws/stt_only")

if __name__ == "__main__":
    main()
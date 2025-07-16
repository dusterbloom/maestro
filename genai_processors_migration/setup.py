from setuptools import setup, find_packages

setup(
    name="maestro-genai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-genai>=0.3.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "websockets>=13.0",
        "aiohttp>=3.10.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
        "pyaudio>=0.2.11",
    ],
    extras_require={
        "stt": ["RealtimeSTT>=0.1.15"],
        "tts": ["pydub>=0.25.1", "librosa>=0.10.0"],
        "analysis": ["torch>=2.0.0", "speechbrain>=0.5.0"],
        "all": ["RealtimeSTT>=0.1.15", "pydub>=0.25.1", "librosa>=0.10.0", "torch>=2.0.0", "speechbrain>=0.5.0"]
    },
    python_requires=">=3.10",
)
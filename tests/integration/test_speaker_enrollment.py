import pytest
from httpx import AsyncClient
from fastapi import FastAPI
import asyncio
import base64
from unittest.mock import patch, MagicMock

# Make sure the app is imported correctly
from orchestrator.src.main import app

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.anyio
async def test_new_user_enrollment_flow(client: AsyncClient):
    """
    Tests the full pipeline for a new user:
    1. First utterance triggers a prompt for an audio sample.
    2. Client sends audio, gets a speaker_id.
    3. Client re-sends first utterance, gets a prompt for a name.
    4. Client sends name, which is stored.
    5. A final utterance is processed normally.
    """
    session_id = "test_enrollment_session"
    initial_transcript = "Hello Maestro"
    user_name = "Tester"
    final_transcript = "How are you?"

    # --- Step 1: First utterance from a new user (no speaker_id) ---
    with patch("orchestrator.src.main.orchestrator.generate_response") as mock_generate_response:
        # Mock the LLM to return the prompt for an audio sample
        mock_generate_response.return_value = "Hello! I don't believe we've spoken before. Could you please provide a 5-second audio sample so I can get to know your voice?"

        response = await client.post("/process-transcript", json={
            "transcript": initial_transcript,
            "session_id": session_id,
            "speaker_id": None
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "prompt_for_embedding" in data and data["prompt_for_embedding"] is True
        assert "provide a 5-second audio sample" in data["response_text"]

    # --- Step 2: Client sends audio for embedding ---
    with patch("orchestrator.src.main.orchestrator.embed_speaker") as mock_embed_speaker:
        mock_speaker_id = "speaker_123"
        mock_embed_speaker.return_value = mock_speaker_id
        
        # Mock audio data (just some bytes)
        mock_audio_b64 = base64.b64encode(b"mock_audio_data").decode('utf-8')
        
        response = await client.post("/embed-speaker", json={"audio_data": mock_audio_b64})
        
        assert response.status_code == 200
        data = response.json()
        assert data["speaker_id"] == mock_speaker_id

    # --- Step 3: Client re-sends initial utterance with new speaker_id ---
    with patch("orchestrator.src.main.orchestrator.generate_response") as mock_generate_response:
        # Mock the LLM to now ask for the user's name
        mock_generate_response.return_value = "Thanks! I've got your voice signature. Now, what would you like me to call you?"

        response = await client.post("/process-transcript", json={
            "transcript": initial_transcript,
            "session_id": session_id,
            "speaker_id": mock_speaker_id
        })

        assert response.status_code == 200
        data = response.json()
        assert "prompt_for_name" in data and data["prompt_for_name"] is True
        assert "what would you like me to call you" in data["response_text"]

    # --- Step 4: Client sends name to be stored ---
    response = await client.post("/set-speaker-name", json={
        "speaker_id": mock_speaker_id,
        "name": user_name
    })
    assert response.status_code == 200
    
    # Verify the name was stored in the orchestrator's in-memory dict
    from orchestrator.src.main import orchestrator
    assert orchestrator._get_speaker_name(mock_speaker_id) == user_name

    # --- Step 5: Final utterance from a now-known user ---
    with patch("orchestrator.src.main.orchestrator.generate_response") as mock_generate_response:
        # Mock the LLM to give a normal response, addressing the user by name
        mock_generate_response.return_value = f"I'm doing great, {user_name}! Thanks for asking."

        response = await client.post("/process-transcript", json={
            "transcript": final_transcript,
            "session_id": session_id,
            "speaker_id": mock_speaker_id
        })

        assert response.status_code == 200
        data = response.json()
        assert data.get("prompt_for_embedding") is not True
        assert data.get("prompt_for_name") is not True
        assert user_name in data["response_text"]

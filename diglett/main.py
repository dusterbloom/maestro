import asyncio
import base64
import io
import json
import logging
from typing import List

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, UploadFile, File
from speechbrain.pretrained import EncoderClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the pre-trained speaker verification model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

@app.post("/embed")
async def create_embedding(file: UploadFile = File(...)):
    """Create a speaker embedding from a 5-second audio file."""
    try:
        audio_data = await file.read()
        signal, fs = torchaudio.load(audio_data)
        
        # Calculate average dB level
        audio_rms = torch.sqrt(torch.mean(signal ** 2))
        avg_db = 20 * torch.log10(audio_rms + 1e-8).item()  # Add small epsilon to avoid log(0)
        
        with torch.no_grad():
            embedding = classifier.encode_batch(signal)[0][0]
        
        # Return format matching what VoiceService expects
        return {
            "speaker_name": "Unknown",  # Default name - will be learned later
            "speaker_embedding": embedding.tolist(),
            "avg_db": avg_db
        }
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return {"error": str(e)}, 500

@app.websocket("/stream")
async def stream_verification(websocket: WebSocket):
    """Real-time speaker verification via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            audio_b64 = message.get("audio_data")
            speaker_embeddings = message.get("speaker_embedding", [])

            if not audio_b64 or not speaker_embeddings:
                await websocket.send_json({"error": "Missing audio or embeddings"})
                continue

            audio_data = base64.b64decode(audio_b64)
            signal, fs = torchaudio.load(audio_data)

            with torch.no_grad():
                live_embedding = classifier.encode_batch(signal)[0][0]

            # Compare with registered speakers
            scores = []
            for emb_list in speaker_embeddings:
                stored_embedding = torch.FloatTensor(emb_list)
                score = torch.nn.functional.cosine_similarity(live_embedding, stored_embedding, dim=0)
                scores.append(score.item())
            
            identified_speaker_index = np.argmax(scores) if scores else -1

            await websocket.send_json({
                "identified_speaker_index": identified_speaker_index,
                "scores": scores
            })

    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        await websocket.close()

#!/usr/bin/env python3
"""
FastAPI server for Mimi audio codec
Provides /encode and /decode endpoints with TRUE HTTP streaming support
"""

import io
import os
import struct
from pathlib import Path
from typing import Optional, AsyncIterator

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse
from transformers import MimiModel, AutoFeatureExtractor
import soundfile as sf

# Constants
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920

# Global model (loaded once at startup)
model: Optional[MimiModel] = None
feature_extractor: Optional[AutoFeatureExtractor] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Mimi Audio Codec API",
    description="Encode audio to tokens and decode tokens to audio",
    version="1.0.0"
)

def load_model():
    """Load the Mimi model at startup."""
    global model, feature_extractor
    
    model_name = os.environ.get("MIMI_MODEL_PATH", "kyutai/mimi")
    print(f"Loading Mimi model from {model_name}...")
    
    # Check for HuggingFace token
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    model = MimiModel.from_pretrained(model_name, token=hf_token)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, token=hf_token)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")

@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

async def encode_stream_generator(audio: np.ndarray) -> AsyncIterator[bytes]:
    """
    Generator that yields encoded tokens progressively as chunks are processed.
    Yields raw binary token data with metadata header.
    """
    # Pad audio to be multiple of frame size
    remainder = len(audio) % SAMPLES_PER_FRAME
    if remainder != 0:
        audio = np.pad(audio, (0, SAMPLES_PER_FRAME - remainder))
    
    num_chunks = len(audio) // SAMPLES_PER_FRAME
    
    # First, yield header: num_chunks (4 bytes)
    yield struct.pack('<I', num_chunks)
    
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * SAMPLES_PER_FRAME
            end = start + SAMPLES_PER_FRAME
            chunk = audio[start:end]
            
            inputs = feature_extractor(
                raw_audio=chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            encoder_outputs = model.encode(inputs["input_values"])
            codes = encoder_outputs.audio_codes.cpu().numpy()
            
            # Yield shape info: (batch, codebooks, frames) - 3 ints
            if i == 0:
                shape_bytes = struct.pack('<III', *codes.shape)
                yield shape_bytes
            
            # Yield raw token data as bytes
            yield codes.astype(np.int32).tobytes()

@app.post("/encode")
async def encode_audio(
    file: UploadFile = File(...),
    streaming: bool = False
):
    """
    Encode audio file to tokens.
    
    Args:
        file: Audio file (wav, mp3, flac, etc.)
        streaming: TRUE HTTP streaming - yields chunks progressively
    
    Returns:
        Binary stream of encoded tokens (custom format) or NPZ file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            import scipy.signal
            audio = scipy.signal.resample_poly(audio, SAMPLE_RATE, sr)
        
        audio = audio.astype(np.float32)
        
        if streaming:
            # TRUE streaming: yield chunks as they are encoded
            return StreamingResponse(
                encode_stream_generator(audio),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": "attachment; filename=tokens.bin",
                    "X-Format": "streaming-binary",
                    "X-Sample-Rate": str(SAMPLE_RATE),
                }
            )
        else:
            # Non-streaming: process all at once, return NPZ
            with torch.no_grad():
                inputs = feature_extractor(
                    raw_audio=audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                encoder_outputs = model.encode(inputs["input_values"])
                audio_codes = encoder_outputs.audio_codes.cpu().numpy()
            
            # Save to NPZ in memory
            npz_buffer = io.BytesIO()
            np.savez_compressed(
                npz_buffer,
                audio_codes=audio_codes,
                sample_rate=SAMPLE_RATE,
                num_codebooks=audio_codes.shape[1],
                num_frames=audio_codes.shape[2],
            )
            npz_buffer.seek(0)
            
            return StreamingResponse(
                npz_buffer,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": "attachment; filename=tokens.npz",
                    "X-Num-Codebooks": str(audio_codes.shape[1]),
                    "X-Num-Frames": str(audio_codes.shape[2]),
                }
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def decode_stream_generator(audio_codes: np.ndarray) -> AsyncIterator[bytes]:
    """
    Generator that yields decoded audio progressively as chunks are processed.
    Yields raw PCM audio data (float32).
    """
    chunk_size = 100
    num_frames = audio_codes.shape[2]
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, num_frames)
            chunk_codes = audio_codes[:, :, start:end]
            
            chunk_codes_tensor = torch.from_numpy(chunk_codes).to(device)
            audio_values = model.decode(chunk_codes_tensor)
            audio_np = audio_values[0].cpu().numpy()
            
            # Ensure 1D
            while audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            
            # Clip to valid range
            audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
            
            # Yield raw PCM audio bytes
            yield audio_np.tobytes()

@app.post("/decode")
async def decode_tokens(
    file: UploadFile = File(...),
    streaming: bool = False
):
    """
    Decode tokens to audio.
    
    Args:
        file: NPZ file with encoded tokens
        streaming: TRUE HTTP streaming - yields audio chunks progressively
    
    Returns:
        Audio stream (raw PCM float32) or WAV file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read NPZ file
        npz_bytes = await file.read()
        data = np.load(io.BytesIO(npz_bytes))
        audio_codes = data["audio_codes"]
        
        if streaming:
            # TRUE streaming: yield audio chunks as they are decoded
            return StreamingResponse(
                decode_stream_generator(audio_codes),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": "attachment; filename=audio.pcm",
                    "X-Format": "raw-pcm-float32",
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Channels": "1",
                }
            )
        else:
            # Non-streaming: process all at once, return WAV
            with torch.no_grad():
                audio_codes_tensor = torch.from_numpy(audio_codes).to(device)
                audio_values = model.decode(audio_codes_tensor)
                audio = audio_values[0].cpu().numpy()
                
                # Ensure 1D
                while audio.ndim > 1:
                    audio = audio.squeeze()
            
            # Clip to valid range
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            # Save to WAV in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, SAMPLE_RATE, format='WAV')
            wav_buffer.seek(0)
            
            return StreamingResponse(
                wav_buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=audio.wav",
                    "X-Duration": str(len(audio) / SAMPLE_RATE),
                }
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Mimi Audio Codec API",
        "version": "1.0.0",
        "endpoints": {
            "/encode": "POST - Encode audio to tokens",
            "/decode": "POST - Decode tokens to audio",
            "/health": "GET - Health check"
        },
        "device": device,
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 6542))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)

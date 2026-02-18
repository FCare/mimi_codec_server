#!/usr/bin/env python3
"""
FastAPI server for Mimi audio codec using ONNX Runtime with Vulkan acceleration
Uses pre-exported ONNX model from https://huggingface.co/onnx-community/kyutai-mimi-ONNX
"""

import io
import os
import struct
from pathlib import Path
from typing import Optional, AsyncIterator

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import soundfile as sf

# Constants
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920

# Global ONNX sessions (loaded once at startup)
encoder_session: Optional[ort.InferenceSession] = None
decoder_session: Optional[ort.InferenceSession] = None

app = FastAPI(
    title="Mimi Audio Codec API (ONNX + Vulkan)",
    description="Encode audio to tokens and decode tokens to audio using ONNX Runtime",
    version="2.0.0-onnx"
)

def load_onnx_models():
    """Load ONNX encoder and decoder models at startup."""
    global encoder_session, decoder_session
    
    model_dir = Path(os.environ.get("MIMI_ONNX_MODEL_PATH", "/app/onnx_model"))
    
    # Use FP16 models to save ~50% RAM (can be changed via env var)
    use_fp16 = os.environ.get("USE_ONNX_FP16", "true").lower() == "true"
    model_suffix = "_fp16.onnx" if use_fp16 else ".onnx"
    
    # Models are in onnx/ subdirectory
    encoder_path = model_dir / "onnx" / f"encoder_model{model_suffix}"
    decoder_path = model_dir / "onnx" / f"decoder_model{model_suffix}"
    
    print(f"Loading ONNX models from {model_dir}/onnx/...")
    print(f"  Precision: {'FP16 (~300MB)' if use_fp16 else 'FP32 (~500MB)'}")
    
    # Configure ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try to use Vulkan ExecutionProvider, fallback to CPU
    # Note: VulkanExecutionProvider may not be available on all platforms
    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")
    
    providers = []
    if 'VulkanExecutionProvider' in available_providers:
        providers.append('VulkanExecutionProvider')
        print("✓ Using Vulkan ExecutionProvider for GPU acceleration")
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
        print("✓ CUDA available as fallback")
    providers.append('CPUExecutionProvider')
    
    # Load encoder
    print(f"Loading encoder from {encoder_path}...")
    encoder_session = ort.InferenceSession(
        str(encoder_path),
        sess_options=sess_options,
        providers=providers
    )
    print(f"  Encoder using: {encoder_session.get_providers()[0]}")
    
    # Load decoder
    print(f"Loading decoder from {decoder_path}...")
    decoder_session = ort.InferenceSession(
        str(decoder_path),
        sess_options=sess_options,
        providers=providers
    )
    print(f"  Decoder using: {decoder_session.get_providers()[0]}")
    
    print("ONNX models loaded successfully")

@app.on_event("startup")
async def startup_event():
    """Load ONNX models when server starts."""
    load_onnx_models()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    encoder_provider = encoder_session.get_providers()[0] if encoder_session else None
    decoder_provider = decoder_session.get_providers()[0] if decoder_session else None
    
    return {
        "status": "healthy",
        "encoder_loaded": encoder_session is not None,
        "decoder_loaded": decoder_session is not None,
        "encoder_provider": encoder_provider,
        "decoder_provider": decoder_provider,
        "runtime": "ONNX Runtime"
    }

async def encode_stream_generator(audio: np.ndarray) -> AsyncIterator[bytes]:
    """
    Generator that yields encoded tokens progressively using ONNX Runtime.
    """
    # Pad audio to be multiple of frame size
    remainder = len(audio) % SAMPLES_PER_FRAME
    if remainder != 0:
        audio = np.pad(audio, (0, SAMPLES_PER_FRAME - remainder))
    
    num_chunks = len(audio) // SAMPLES_PER_FRAME
    
    # First, yield header: num_chunks (4 bytes)
    yield struct.pack('<I', num_chunks)
    
    for i in range(num_chunks):
        start = i * SAMPLES_PER_FRAME
        end = start + SAMPLES_PER_FRAME
        chunk = audio[start:end]
        
        # Prepare input for ONNX (shape: [batch, channels, samples])
        input_values = chunk.astype(np.float32).reshape(1, 1, -1)
        
        # Run ONNX encoder
        ort_inputs = {encoder_session.get_inputs()[0].name: input_values}
        ort_outputs = encoder_session.run(None, ort_inputs)
        
        # Get audio codes (first output)
        codes = ort_outputs[0]  # Shape: [batch, codebooks, frames]
        
        # Yield shape info on first chunk
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
    Encode audio file to tokens using ONNX Runtime.
    
    Args:
        file: Audio file (wav, mp3, flac, etc.)
        streaming: TRUE HTTP streaming - yields chunks progressively
    
    Returns:
        Binary stream of encoded tokens (custom format) or NPZ file
    """
    if encoder_session is None:
        raise HTTPException(status_code=503, detail="Encoder model not loaded")
    
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
            # Non-streaming: process all at once
            input_values = audio.reshape(1, 1, -1)
            ort_inputs = {encoder_session.get_inputs()[0].name: input_values}
            ort_outputs = encoder_session.run(None, ort_inputs)
            audio_codes = ort_outputs[0]
            
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
    Generator that yields decoded audio progressively using ONNX Runtime.
    """
    chunk_size = 100
    num_frames = audio_codes.shape[2]
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, num_frames)
        chunk_codes = audio_codes[:, :, start:end]
        
        # Run ONNX decoder
        ort_inputs = {decoder_session.get_inputs()[0].name: chunk_codes}
        ort_outputs = decoder_session.run(None, ort_inputs)
        
        audio_np = ort_outputs[0][0]  # Remove batch dimension
        
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
    Decode tokens to audio using ONNX Runtime.
    
    Args:
        file: NPZ file with encoded tokens
        streaming: TRUE HTTP streaming - yields audio chunks progressively
    
    Returns:
        Audio stream (raw PCM float32) or WAV file
    """
    if decoder_session is None:
        raise HTTPException(status_code=503, detail="Decoder model not loaded")
    
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
            # Non-streaming: process all at once
            ort_inputs = {decoder_session.get_inputs()[0].name: audio_codes}
            ort_outputs = decoder_session.run(None, ort_inputs)
            audio = ort_outputs[0][0]
            
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
    encoder_provider = encoder_session.get_providers()[0] if encoder_session else "not loaded"
    decoder_provider = decoder_session.get_providers()[0] if decoder_session else "not loaded"
    
    return {
        "name": "Mimi Audio Codec API (ONNX)",
        "version": "2.0.0-onnx",
        "runtime": "ONNX Runtime",
        "endpoints": {
            "/encode": "POST - Encode audio to tokens",
            "/decode": "POST - Decode tokens to audio",
            "/health": "GET - Health check"
        },
        "encoder_provider": encoder_provider,
        "decoder_provider": decoder_provider,
        "models_loaded": encoder_session is not None and decoder_session is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 6543))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)

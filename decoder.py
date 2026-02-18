#!/usr/bin/env python3
"""
Mimi Audio Decoder - Decode discrete tokens to audio using Mimi codec
Uses the same parameters as unmute project (SAMPLE_RATE=24000, FRAME_SIZE=1920)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import MimiModel

# Fix SSL certificate issues if needed
def setup_ssl_fix():
    """Setup to handle SSL certificate issues."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

# Constants matching unmute configuration
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920
FRAME_TIME_SEC = SAMPLES_PER_FRAME / SAMPLE_RATE  # 0.08 seconds

def save_audio(audio: np.ndarray, output_path: Path, sample_rate: int = SAMPLE_RATE):
    """Save audio array to file."""
    try:
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
    except ImportError:
        raise ImportError(
            "soundfile is required for audio saving. "
            "Install with: pip install soundfile"
        )

def decode_tokens(
    tokens_path: Path,
    output_path: Path,
    model_name: str = "kyutai/mimi",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    streaming: bool = False,
    trust_remote_code: bool = False,
):
    """
    Decode tokens to audio using Mimi model.
    
    Args:
        tokens_path: Path to input tokens file (.npz)
        output_path: Path to save decoded audio
        model_name: HuggingFace model identifier
        device: Device to run inference on ('cuda' or 'cpu')
        streaming: If True, process tokens in chunks (for long sequences)
    """
    print(f"Loading Mimi model from {model_name}...")
    
    # Check for HuggingFace token
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    if hf_token:
        print("Using HuggingFace token from environment variable")
    
    try:
        model = MimiModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. SSL Certificate error: Try running with --fix-ssl")
        print("2. Model not found: Check the model name")
        print("3. No internet: Download the model manually and use the local path")
        print("4. Missing token: Set HUGGING_FACE_HUB_TOKEN environment variable")
        raise
    model = model.to(device)
    model.eval()
    
    print(f"Loading tokens from {tokens_path}...")
    data = np.load(tokens_path)
    audio_codes = data["audio_codes"]
    
    # Extract metadata
    stored_sample_rate = int(data.get("sample_rate", SAMPLE_RATE))
    num_codebooks = int(data.get("num_codebooks", audio_codes.shape[1]))
    num_frames = int(data.get("num_frames", audio_codes.shape[2]))
    
    print(f"Token shape: {audio_codes.shape}")
    print(f"Number of codebooks: {num_codebooks}")
    print(f"Number of frames: {num_frames}")
    print(f"Expected duration: {num_frames / 12.5:.2f}s (at 12.5 fps)")
    
    if stored_sample_rate != SAMPLE_RATE:
        print(f"Warning: Stored sample rate ({stored_sample_rate}) differs from expected ({SAMPLE_RATE})")
    
    with torch.no_grad():
        if streaming:
            # Process in chunks for long token sequences
            print("Processing in streaming mode...")
            all_audio = []
            
            # Process in chunks (e.g., 100 frames at a time)
            chunk_size = 100
            num_chunks = (num_frames + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, num_frames)
                chunk_codes = audio_codes[:, :, start:end]
                
                # Convert to tensor
                chunk_codes_tensor = torch.from_numpy(chunk_codes).to(device)
                
                # Decode
                audio_values = model.decode(chunk_codes_tensor)
                audio_np = audio_values[0].cpu().numpy()
                
                # audio_np shape is typically [channels, samples]
                if audio_np.ndim > 1:
                    audio_np = audio_np[0]  # Take first channel (mono)
                
                all_audio.append(audio_np)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{num_chunks} chunks...")
            
            # Concatenate all audio chunks
            audio = np.concatenate(all_audio)
        else:
            # Process all tokens at once
            print("Processing all tokens...")
            audio_codes_tensor = torch.from_numpy(audio_codes).to(device)
            
            # Decode to audio
            audio_values = model.decode(audio_codes_tensor)
            audio = audio_values[0].cpu().numpy()
            
            # audio shape is typically [channels, samples]
            if audio.ndim > 1:
                audio = audio[0]  # Take first channel (mono)
    
    # Ensure audio is 1D
    while audio.ndim > 1:
        audio = audio.squeeze()
    
    print(f"Decoded audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f}s")
    
    # Clip to [-1, 1] range
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    
    # Save audio
    save_audio(audio, output_path, SAMPLE_RATE)
    print(f"Saved decoded audio to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Decode discrete tokens to audio using Mimi codec"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input tokens file (.npz)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output audio file (.wav)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kyutai/mimi",
        help="HuggingFace model identifier (default: kyutai/mimi)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Process tokens in chunks (for very long sequences)"
    )
    parser.add_argument(
        "--fix-ssl",
        action="store_true",
        help="Fix SSL certificate verification issues (use with caution)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace (required for some models)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output has audio extension
    if args.output.suffix not in [".wav", ".flac"]:
        args.output = args.output.with_suffix(".wav")
    
    if args.fix_ssl:
        print("WARNING: Disabling SSL certificate verification")
        setup_ssl_fix()
    
    try:
        decode_tokens(
            args.input,
            args.output,
            model_name=args.model,
            device=args.device,
            streaming=args.streaming,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        print(f"Error during decoding: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

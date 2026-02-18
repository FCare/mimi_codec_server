#!/usr/bin/env python3
"""
Mimi Audio Encoder - Encode audio to discrete tokens using Mimi codec
Uses the same parameters as unmute project (SAMPLE_RATE=24000, FRAME_SIZE=1920)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import MimiModel, AutoFeatureExtractor

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

def load_audio(audio_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != target_sr:
            import scipy.signal
            audio = scipy.signal.resample_poly(
                audio, target_sr, sr
            )
        
        return audio.astype(np.float32)
    except ImportError:
        raise ImportError(
            "soundfile is required for audio loading. "
            "Install with: pip install soundfile"
        )

def encode_audio(
    audio_path: Path,
    output_path: Path,
    model_name: str = "kyutai/mimi",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    streaming: bool = False,
    trust_remote_code: bool = False,
):
    """
    Encode audio file to tokens using Mimi model.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to save encoded tokens (.npz file)
        model_name: HuggingFace model identifier
        device: Device to run inference on ('cuda' or 'cpu')
        streaming: If True, process audio in chunks (for long files)
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
        feature_extractor = AutoFeatureExtractor.from_pretrained(
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
    
    print(f"Loading audio from {audio_path}...")
    audio = load_audio(audio_path, target_sr=SAMPLE_RATE)
    
    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f}s")
    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    
    with torch.no_grad():
        if streaming:
            # Process in chunks for long audio files
            print("Processing in streaming mode...")
            all_codes = []
            
            # Pad audio to be multiple of frame size
            remainder = len(audio) % SAMPLES_PER_FRAME
            if remainder != 0:
                audio = np.pad(audio, (0, SAMPLES_PER_FRAME - remainder))
            
            num_chunks = len(audio) // SAMPLES_PER_FRAME
            
            for i in range(num_chunks):
                start = i * SAMPLES_PER_FRAME
                end = start + SAMPLES_PER_FRAME
                chunk = audio[start:end]
                
                # Pre-process the chunk
                inputs = feature_extractor(
                    raw_audio=chunk,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Encode
                encoder_outputs = model.encode(inputs["input_values"])
                codes = encoder_outputs.audio_codes.cpu().numpy()
                all_codes.append(codes)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{num_chunks} chunks...")
            
            # Concatenate all codes
            audio_codes = np.concatenate(all_codes, axis=-1)
        else:
            # Process entire audio at once
            print("Processing entire audio...")
            inputs = feature_extractor(
                raw_audio=audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Encode to discrete codes
            encoder_outputs = model.encode(inputs["input_values"])
            audio_codes = encoder_outputs.audio_codes.cpu().numpy()
    
    # Save tokens
    print(f"Encoded shape: {audio_codes.shape}")
    print(f"Number of codebooks: {audio_codes.shape[1]}")
    print(f"Number of frames: {audio_codes.shape[2]}")
    print(f"Frame rate: {audio_codes.shape[2] / (len(audio) / SAMPLE_RATE):.2f} Hz")
    
    np.savez_compressed(
        output_path,
        audio_codes=audio_codes,
        sample_rate=SAMPLE_RATE,
        num_codebooks=audio_codes.shape[1],
        num_frames=audio_codes.shape[2],
    )
    print(f"Saved encoded tokens to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Encode audio to discrete tokens using Mimi codec"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input audio file (wav, mp3, flac, etc.)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output file (.npz)"
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
        help="Process audio in chunks (for very long files)"
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
    
    # Ensure output has .npz extension
    if args.output.suffix != ".npz":
        args.output = args.output.with_suffix(".npz")
    
    if args.fix_ssl:
        print("WARNING: Disabling SSL certificate verification")
        setup_ssl_fix()
    
    try:
        encode_audio(
            args.input,
            args.output,
            model_name=args.model,
            device=args.device,
            streaming=args.streaming,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        print(f"Error during encoding: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

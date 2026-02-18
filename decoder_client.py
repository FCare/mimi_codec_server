#!/usr/bin/env python3
"""
Mimi Audio Decoder Client - Calls the FastAPI server to decode tokens
Supports TRUE HTTP streaming with progressive chunk reception
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

def decode_tokens_via_api(
    tokens_path: Path,
    output_path: Path,
    api_url: str = "http://localhost:8000",
    streaming: bool = False,
):
    """
    Decode tokens to audio via API.
    
    Args:
        tokens_path: Path to input tokens file (.npz)
        output_path: Path to save decoded audio
        api_url: Base URL of the Mimi API server
        streaming: If True, TRUE HTTP streaming - receives audio chunks progressively
    """
    print(f"Decoding {tokens_path} via API at {api_url}...")
    if streaming:
        print("  Mode: TRUE HTTP streaming (progressive chunks)")
    
    # Prepare the file upload
    with open(tokens_path, 'rb') as f:
        files = {'file': (tokens_path.name, f, 'application/octet-stream')}
        params = {'streaming': str(streaming).lower()}
        
        # Send request with stream=True to receive chunks progressively
        response = requests.post(
            f"{api_url}/decode",
            files=files,
            params=params,
            timeout=300,
            stream=True  # Enable progressive streaming
        )
    
    if response.status_code == 200:
        if streaming:
            # Streaming mode: receive PCM chunks progressively
            print("  Receiving audio chunks progressively...")
            
            sample_rate = int(response.headers.get('X-Sample-Rate', 24000))
            all_audio = []
            chunks_received = 0
            bytes_received = 0
            
            # Read PCM chunks progressively
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    # Convert bytes to float32 PCM
                    audio_chunk = np.frombuffer(chunk, dtype=np.float32)
                    all_audio.append(audio_chunk)
                    chunks_received += 1
                    bytes_received += len(chunk)
                    print(f"  ✓ Chunk {chunks_received} received ({len(audio_chunk)} samples)", end='\r')
            
            print(f"\n  Total bytes received: {bytes_received}")
            
            # Concatenate all chunks
            audio = np.concatenate(all_audio)
            duration = len(audio) / sample_rate
            
            # Save to WAV
            sf.write(output_path, audio, sample_rate)
            
            print(f"✓ Decoding successful (streaming)!")
            print(f"  Chunks received: {chunks_received}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Samples: {len(audio)}")
            print(f"  Saved to: {output_path}")
        else:
            # Non-streaming mode: save WAV directly
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            duration = response.headers.get('X-Duration', 'unknown')
            
            print(f"✓ Decoding successful!")
            print(f"  Duration: {duration}s")
            print(f"  Saved to: {output_path}")
    else:
        print(f"✗ Decoding failed!")
        print(f"  Status code: {response.status_code}")
        print(f"  Error: {response.text}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Decode tokens to audio using Mimi API server"
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
        "--api-url",
        type=str,
        default="http://localhost:6542",
        help="Base URL of the Mimi API server"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Process tokens in chunks (for very long sequences)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output has audio extension
    if args.output.suffix not in [".wav", ".flac"]:
        args.output = args.output.with_suffix(".wav")
    
    try:
        decode_tokens_via_api(
            args.input,
            args.output,
            api_url=args.api_url,
            streaming=args.streaming,
        )
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API server at {args.api_url}", file=sys.stderr)
        print("Make sure the server is running with: python server.py", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during decoding: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

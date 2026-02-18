#!/usr/bin/env python3
"""
Mimi Audio Encoder Client - Calls the FastAPI server to encode audio
Supports TRUE HTTP streaming with progressive chunk reception
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import requests

def encode_audio_via_api(
    audio_path: Path,
    output_path: Path,
    api_url: str = "http://localhost:8000",
    streaming: bool = False,
):
    """
    Encode audio file to tokens via API.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to save encoded tokens
        api_url: Base URL of the Mimi API server
        streaming: If True, TRUE HTTP streaming - receives chunks progressively
    """
    print(f"Encoding {audio_path} via API at {api_url}...")
    if streaming:
        print("  Mode: TRUE HTTP streaming (progressive chunks)")
    
    # Prepare the file upload
    with open(audio_path, 'rb') as f:
        files = {'file': (audio_path.name, f, 'audio/wav')}
        params = {'streaming': str(streaming).lower()}
        
        # Send request with stream=True to receive chunks progressively
        response = requests.post(
            f"{api_url}/encode",
            files=files,
            params=params,
            timeout=300,
            stream=True  # Enable progressive streaming
        )
    
    if response.status_code == 200:
        if streaming:
            # Streaming mode: receive binary chunks progressively
            print("  Receiving chunks progressively...")
            
            all_tokens = []
            chunks_received = 0
            bytes_received = 0
            
            # Create single iterator for sequential reading
            stream_iter = response.iter_content(chunk_size=8192)
            buffer = b''
            
            # Helper to read exact number of bytes from buffer + iterator
            def read_exact(n):
                nonlocal buffer
                while len(buffer) < n:
                    chunk = next(stream_iter, None)
                    if chunk is None:
                        raise ValueError(f"Stream ended, need {n} bytes but only have {len(buffer)}")
                    buffer += chunk
                result = buffer[:n]
                buffer = buffer[n:]
                return result
            
            # Read header: num_chunks (4 bytes)
            header_data = read_exact(4)
            num_chunks = struct.unpack('<I', header_data)[0]
            print(f"  Expected chunks: {num_chunks}")
            
            # Read shape (12 bytes: 3 ints)
            shape_data = read_exact(12)
            batch, codebooks, frames_per_chunk = struct.unpack('<III', shape_data)
            print(f"  Token shape per chunk: ({batch}, {codebooks}, {frames_per_chunk})")
            
            # Calculate expected bytes per chunk
            bytes_per_chunk = batch * codebooks * frames_per_chunk * 4  # int32
            
            # Read token chunks progressively
            for i in range(num_chunks):
                token_bytes = read_exact(bytes_per_chunk)
                bytes_received += bytes_per_chunk
                
                tokens = np.frombuffer(token_bytes, dtype=np.int32).reshape(batch, codebooks, frames_per_chunk)
                all_tokens.append(tokens)
                chunks_received += 1
                print(f"  ✓ Chunk {chunks_received}/{num_chunks} received", end='\r')
            
            print(f"\n  Total bytes received: {bytes_received}")
            
            # Concatenate all chunks
            audio_codes = np.concatenate(all_tokens, axis=-1)
            
            # Save to NPZ
            np.savez_compressed(
                output_path,
                audio_codes=audio_codes,
                sample_rate=24000,
                num_codebooks=codebooks,
                num_frames=audio_codes.shape[2],
            )
            
            print(f"✓ Encoding successful (streaming)!")
            print(f"  Chunks received: {chunks_received}")
            print(f"  Final shape: {audio_codes.shape}")
            print(f"  Saved to: {output_path}")
        else:
            # Non-streaming mode: save NPZ directly
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            num_codebooks = response.headers.get('X-Num-Codebooks', 'unknown')
            num_frames = response.headers.get('X-Num-Frames', 'unknown')
            
            print(f"✓ Encoding successful!")
            print(f"  Number of codebooks: {num_codebooks}")
            print(f"  Number of frames: {num_frames}")
            print(f"  Saved to: {output_path}")
    else:
        print(f"✗ Encoding failed!")
        print(f"  Status code: {response.status_code}")
        print(f"  Error: {response.text}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Encode audio to tokens using Mimi API server"
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
        "--api-url",
        type=str,
        default="http://localhost:6542",
        help="Base URL of the Mimi API server"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Process audio in chunks (for very long files)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output has .npz extension
    if args.output.suffix != ".npz":
        args.output = args.output.with_suffix(".npz")
    
    try:
        encode_audio_via_api(
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
        print(f"Error during encoding: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

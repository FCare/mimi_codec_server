# Mimi Audio Codec - PyTorch Implementation

Pure PyTorch implementation of the Mimi audio codec for encoding and decoding audio using discrete tokens. This implementation uses the official `transformers.MimiModel` and matches the exact parameters used in the [unmute project](https://github.com/kyutai-labs/unmute).

## Features

- ✅ **PyTorch-based**: Uses `transformers.MimiModel` (pure PyTorch, no dependency on the `moshi` library)
- ✅ **Same parameters as unmute**: Sample rate 24kHz, frame size 1920 samples
- ✅ **Simple CLI**: Easy-to-use command-line interface
- ✅ **Streaming support**: Process long audio files in chunks
- ✅ **GPU acceleration**: Automatic CUDA support when available

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch transformers numpy soundfile scipy
```

## Configuration

The implementation uses the same parameters as the unmute project:

- **Sample Rate**: 24,000 Hz
- **Frame Size**: 1,920 samples
- **Frame Duration**: 0.08 seconds (80ms)
- **Model**: `kyutai/mimi` from HuggingFace

## Quick Start

### First Time Setup

**Step 1: Set HuggingFace Token (Recommended)**

Even though the model is public, a token helps avoid rate limits:

```bash
# Get your token from https://huggingface.co/settings/tokens
export HUGGING_FACE_HUB_TOKEN=your_token_here

# Or use HF_TOKEN
export HF_TOKEN=your_token_here
```

**Step 2: Handle SSL Issues (if needed)**

If you encounter SSL certificate errors (common on macOS or corporate networks):

```bash
# Option A: Try with SSL fix
python encoder.py input.wav tokens.npz --fix-ssl

# Option B: Download the model manually first
git lfs install
git clone https://huggingface.co/kyutai/mimi
python encoder.py input.wav tokens.npz --model ./mimi
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more solutions.

## Usage

### Encoding Audio to Tokens

Convert audio files to discrete token representations:

```bash
# Basic usage
python encoder.py input.wav tokens.npz

# Use CPU instead of GPU
python encoder.py input.wav tokens.npz --device cpu

# Process long files in streaming mode
python encoder.py long_audio.wav tokens.npz --streaming

# Use a different model or local path
python encoder.py input.wav tokens.npz --model kyutai/mimi

# Fix SSL certificate issues
python encoder.py input.wav tokens.npz --fix-ssl
```

**Output**: Creates a compressed `.npz` file containing:
- `audio_codes`: The discrete tokens (shape: [batch, num_codebooks, num_frames])
- `sample_rate`: Sample rate used (24000)
- `num_codebooks`: Number of codebook levels
- `num_frames`: Number of encoded frames

### Decoding Tokens to Audio

Reconstruct audio from discrete tokens:

```bash
# Basic usage
python decoder.py tokens.npz output.wav

# Use CPU instead of GPU
python decoder.py tokens.npz output.wav --device cpu

# Process long sequences in streaming mode
python decoder.py tokens.npz output.wav --streaming

# Use a different model or local path
python decoder.py tokens.npz output.wav --model kyutai/mimi

# Fix SSL certificate issues
python decoder.py tokens.npz output.wav --fix-ssl
```

**Output**: Creates a `.wav` file with the reconstructed audio at 24kHz.

### Round-trip Example

```bash
# Encode
python encoder.py original.wav tokens.npz

# Decode
python decoder.py tokens.npz reconstructed.wav

# Compare (they should be very similar but not identical due to lossy compression)
```

## Python API Usage

### Encoding

```python
from pathlib import Path
from encoder import encode_audio

encode_audio(
    audio_path=Path("input.wav"),
    output_path=Path("tokens.npz"),
    model_name="kyutai/mimi",
    device="cuda",  # or "cpu"
    streaming=False
)
```

### Decoding

```python
from pathlib import Path
from decoder import decode_tokens

decode_tokens(
    tokens_path=Path("tokens.npz"),
    output_path=Path("output.wav"),
    model_name="kyutai/mimi",
    device="cuda",  # or "cpu"
    streaming=False
)
```

### Low-level PyTorch Usage

```python
import torch
import numpy as np
from transformers import MimiModel, AutoFeatureExtractor

# Load model
model = MimiModel.from_pretrained("kyutai/mimi").cuda()
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

# Load audio (example with random data)
audio = np.random.randn(48000).astype(np.float32)  # 2 seconds at 24kHz

# Pre-process
inputs = feature_extractor(
    raw_audio=audio,
    sampling_rate=24000,
    return_tensors="pt"
)
inputs = {k: v.cuda() for k, v in inputs.items()}

# Encode
with torch.no_grad():
    encoder_outputs = model.encode(inputs["input_values"])
    audio_codes = encoder_outputs.audio_codes  # Shape: [1, num_codebooks, num_frames]

# Decode
with torch.no_grad():
    audio_values = model.decode(audio_codes)  # Shape: [1, 1, num_samples]
    reconstructed = audio_values[0, 0].cpu().numpy()
```

## Technical Details

### Mimi Model

Mimi is a state-of-the-art neural audio codec developed by Kyutai. It uses:

- **Encoder-Decoder Architecture**: SEANet-based encoder and decoder
- **Residual Vector Quantization**: Multiple codebook levels for high-quality reconstruction
- **Transformer**: Optional transformer layers for better context modeling
- **Frame Rate**: 12.5 Hz (one frame every 80ms)
- **Compression**: ~12.5 tokens/second × num_codebooks

### Token Format

The encoded tokens are stored in a NumPy `.npz` file with the following structure:

```python
import numpy as np

data = np.load("tokens.npz")
audio_codes = data["audio_codes"]  # Shape: [batch=1, num_codebooks, num_frames]
sample_rate = data["sample_rate"]  # 24000
num_codebooks = data["num_codebooks"]  # Typically 8, 16, or 32
num_frames = data["num_frames"]  # Number of temporal frames
```

### Performance

- **Encoding**: ~100-500× realtime on GPU (depends on GPU and audio length)
- **Decoding**: ~100-500× realtime on GPU
- **Memory**: ~2GB VRAM for the model + audio buffer
- **File Size**: ~12.5 KB/second of audio (with 8 codebooks, int16 storage)

## Comparison with Unmute

This implementation is designed to be compatible with the [unmute project](https://github.com/kyutai-labs/unmute):

| Feature | This Implementation | Unmute |
|---------|-------------------|--------|
| Backend | `transformers.MimiModel` (PyTorch) | Custom Rust/Python wrapper |
| Sample Rate | 24,000 Hz | 24,000 Hz |
| Frame Size | 1,920 samples | 1,920 samples |
| Model Weights | `kyutai/mimi` (HuggingFace) | Same weights, different loader |
| Dependencies | PyTorch + Transformers | moshi library |

## Common Issues

### SSL Certificate Errors

If you see `[SSL: CERTIFICATE_VERIFY_FAILED]`:
```bash
python encoder.py input.wav tokens.npz --fix-ssl
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more details and permanent fixes.

### CUDA Out of Memory

If you encounter OOM errors with long audio files:

```bash
# Use streaming mode
python encoder.py long_audio.wav tokens.npz --streaming

# Or use CPU
python encoder.py long_audio.wav tokens.npz --device cpu
```

### Sample Rate Mismatch

The model expects 24kHz audio. If your input is a different sample rate, it will be automatically resampled.

### Model Download

On first run, the model will be downloaded from HuggingFace (~300MB). It will be cached in `~/.cache/huggingface/`.

## References

- [Mimi Model Card](https://huggingface.co/kyutai/mimi)
- [Moshi Project](https://github.com/kyutai-labs/moshi)
- [Unmute Project](https://github.com/kyutai-labs/unmute)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## License

This implementation follows the licenses of the underlying models and libraries:
- Mimi model: See [model card](https://huggingface.co/kyutai/mimi)
- Transformers: Apache 2.0
- PyTorch: BSD-style license

## Contributing

Contributions are welcome! Please ensure:
- Code follows the same style as existing files
- All parameters match the unmute configuration
- Tests pass (if applicable)

# Troubleshooting Guide

## SSL Certificate Errors

### Symptom
```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

### Solutions

**Option 1: Use the --fix-ssl flag (Quick fix)**
```bash
python encoder.py input.wav tokens.npz --fix-ssl
```

⚠️ **Warning**: This disables SSL verification. Use only if you trust your network.

**Option 2: Fix certificates properly (Recommended)**

On macOS:
```bash
# Install certificates
/Applications/Python\ 3.12/Install\ Certificates.command
```

On Linux (Ubuntu/Debian):
```bash
sudo apt-get install ca-certificates
sudo update-ca-certificates
```

On Linux (if using Conda):
```bash
conda install -c conda-forge ca-certificates
```

**Option 3: Download model manually**

1. Download the model from HuggingFace:
   ```bash
   # Using git-lfs
   git lfs install
   git clone https://huggingface.co/kyutai/mimi
   ```

2. Use the local path:
   ```bash
   python encoder.py input.wav tokens.npz --model ./mimi
   ```

**Option 4: Set environment variable**
```bash
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
python encoder.py input.wav tokens.npz
```

## Model Not Found

### Symptom
```
Can't load the configuration of 'kyutai/mimi'
```

### Solutions

1. **Set HuggingFace token**
   
   Even though `kyutai/mimi` is a public model, having a token can help avoid rate limits:
   
   ```bash
   # Set the token
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   # or
   export HF_TOKEN=your_token_here
   
   # Then run the script
   python encoder.py input.wav tokens.npz
   ```
   
   Get your token from: https://huggingface.co/settings/tokens

2. **Check internet connection**
   ```bash
   ping huggingface.co
   ```

3. **Verify model name**
   The correct model name is `kyutai/mimi`. Make sure you're using this exact name.

4. **Download manually** (see SSL section Option 3 above)

5. **Check HuggingFace Hub status**
   Visit https://status.huggingface.co/

## CUDA Out of Memory

### Symptom
```
CUDA out of memory
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

### Solutions

**Option 1: Use CPU**
```bash
python encoder.py input.wav tokens.npz --device cpu
```

**Option 2: Use streaming mode**
```bash
python encoder.py long_audio.wav tokens.npz --streaming
```

**Option 3: Process in smaller chunks**
Split your audio into smaller files first, then process them separately.

**Option 4: Clear GPU memory**
```python
import torch
torch.cuda.empty_cache()
```

## Audio Loading Issues

### Symptom
```
ImportError: soundfile is required for audio loading
```

### Solution
```bash
pip install soundfile

# On some systems you might also need libsndfile
# Ubuntu/Debian:
sudo apt-get install libsndfile1

# macOS:
brew install libsndfile
```

## Transformers Version Issues

### Symptom
```
AttributeError: module 'transformers' has no attribute 'MimiModel'
```

### Solution
Update transformers to the latest version:
```bash
pip install --upgrade transformers
```

The MimiModel was added in transformers >= 4.35.0.

## Sample Rate Mismatch

### Symptom
Audio sounds too fast or too slow after reconstruction.

### Solution
The Mimi model expects 24kHz audio. Make sure your input is at this sample rate, or let the script handle resampling automatically (it should do this by default).

To check your audio's sample rate:
```python
import soundfile as sf
info = sf.info('your_audio.wav')
print(f"Sample rate: {info.samplerate}")
```

## Permission Denied

### Symptom
```
PermissionError: [Errno 13] Permission denied
```

### Solutions

1. **Check file permissions**
   ```bash
   ls -l input.wav
   chmod 644 input.wav  # if needed
   ```

2. **Check output directory permissions**
   ```bash
   ls -ld output_directory/
   chmod 755 output_directory/  # if needed
   ```

3. **Run without sudo** (don't use sudo with these scripts)

## Import Errors

### Symptom
```
ModuleNotFoundError: No module named 'XXX'
```

### Solution
Install all dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch transformers numpy soundfile scipy
```

## Performance Issues

### Slow Processing

1. **Use GPU if available**
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   
   # If True, the scripts will use GPU by default
   # If False, explicitly use CPU:
   python encoder.py input.wav tokens.npz --device cpu
   ```

2. **Use streaming for large files**
   ```bash
   python encoder.py large_file.wav tokens.npz --streaming
   ```

### Memory Usage

Monitor memory usage:
```bash
# On Linux:
watch -n 1 nvidia-smi  # GPU memory
htop  # System memory

# On macOS:
Activity Monitor (GUI)
```

## Getting Help

If you're still having issues:

1. **Check the logs**: Look at the full error traceback
2. **Verify your setup**:
   ```bash
   python --version  # Should be 3.8+
   pip list | grep -E "torch|transformers|numpy"
   ```
3. **Try the test script**:
   ```bash
   python test_mimi.py
   ```
4. **Create a minimal reproduction**:
   ```python
   from transformers import MimiModel
   model = MimiModel.from_pretrained("kyutai/mimi")
   print("Model loaded successfully!")
   ```

## Common Workflow Issues

### Encoder works but decoder fails

Make sure the token file (.npz) wasn't corrupted:
```python
import numpy as np
data = np.load("tokens.npz")
print(data.files)  # Should show: ['audio_codes', 'sample_rate', 'num_codebooks', 'num_frames']
print(data['audio_codes'].shape)  # Should be (1, num_codebooks, num_frames)
```

### Quality is poor

1. Check input audio quality (should be high quality, 24kHz)
2. Verify the model loaded correctly
3. Check for clipping in the input audio
4. Try with a known good audio file (like the test script generates)

### File size too large

The .npz files are compressed. If they're still large:
1. This is expected for long audio files
2. Approximate size: ~12.5 KB per second of audio (with 8 codebooks)
3. Use streaming mode to process in chunks if needed

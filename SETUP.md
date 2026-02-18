# Setup Guide - Mimi Codec Server

## Prerequisites

- Docker and Docker Compose installed
- Git with Git LFS support
- ~300MB disk space for the model

## One-Time Setup

### Step 1: Download the Mimi Model

The model needs to be downloaded **once** on your host machine:

```bash
# Install git-lfs if not already installed
git lfs install

# Clone the model repository (this will be mounted into Docker)
git clone https://huggingface.co/kyutai/mimi model

# Verify the model was downloaded
ls -lh model/
# Should show: config.json, model.safetensors, preprocessor_config.json, README.md
```

The model will be downloaded to `./model/` and is approximately **300MB**.

### Step 2: Build and Start the Server

```bash
# Build the Docker image
docker compose build

# Start the server in detached mode
docker compose up -d

# Check logs
docker compose logs -f

# You should see:
# "Model loaded successfully on cpu"
# "Application startup complete"
```

The server is now running on **http://localhost:6542**

## Verification

Test that everything works:

```bash
# Check health endpoint
curl http://localhost:6542/health

# Should return:
# {"status":"healthy","model_loaded":true,"device":"cpu"}

# Test encode (requires an audio file)
python encoder_client.py test.wav tokens.npz

# Test decode
python decoder_client.py tokens.npz output.wav
```

## Directory Structure

After setup, your directory should look like:

```
mimi_codec/
├── model/                    # Mimi model (git cloned, mounted read-only in Docker)
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── README.md
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Docker image definition
├── server.py                 # FastAPI server
├── encoder_client.py         # Client to encode
├── decoder_client.py         # Client to decode
├── encoder.py                # Standalone encoder (local)
├── decoder.py                # Standalone decoder (local)
└── requirements_server.txt   # Python dependencies
```

## Model Reuse

The `model/` directory is **persistent** on your host:
- Survives `docker compose down`
- Survives image rebuilds
- Can be shared across multiple projects
- Mounted **read-only** in Docker for safety

To use a different model location:

```bash
# Edit docker-compose.yml, change:
# - ./model:/app/model:ro
# to:
# - /path/to/your/model:/app/model:ro
```

## Updating the Model

If a new version of Mimi is released:

```bash
# Stop the server
docker compose down

# Update the model
cd model
git pull
cd ..

# Restart the server
docker compose up -d
```

## Troubleshooting

### Model not found error

```
Error: Can't load the configuration of '/app/model'
```

**Solution**: Make sure you downloaded the model first:
```bash
git clone https://huggingface.co/kyutai/mimi model
```

### Permission denied

```
Error: Permission denied: '/app/model/config.json'
```

**Solution**: Fix permissions:
```bash
chmod -R 755 model/
```

### Git LFS not installed

```
Error: This repository is configured for Git LFS but 'git-lfs' was not found
```

**Solution**: Install git-lfs:
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Then run
git lfs install
git lfs pull
```

## Next Steps

- Read [API_README.md](API_README.md) for API documentation
- Read [README.md](README.md) for standalone usage
- Visit http://localhost:6542/docs for interactive API docs

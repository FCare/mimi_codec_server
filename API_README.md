# Mimi Audio Codec API Server

FastAPI server for encoding and decoding audio using the Mimi codec with streaming support.

## Features

- ✅ **FastAPI**: Modern, fast HTTP API
- ✅ **Streaming**: Support for long audio files and sequences
- ✅ **Docker**: Containerized deployment with GPU support
- ✅ **Health checks**: Built-in health monitoring
- ✅ **Client scripts**: Easy-to-use Python clients

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and start the server
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the server
docker-compose down
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements_server.txt

# Set HuggingFace token (optional)
export HUGGING_FACE_HUB_TOKEN=your_token_here

# Run the server
python server.py

# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 6542
```

The server will be available at `http://localhost:6542`

## API Documentation

Once the server is running, visit:
- **Interactive docs**: http://localhost:6542/docs
- **Alternative docs**: http://localhost:6542/redoc
- **OpenAPI JSON**: http://localhost:6542/openapi.json

## API Endpoints

### GET /health

Health check endpoint.

```bash
curl http://localhost:6542/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### POST /encode

Encode audio to tokens.

```bash
# Using curl
curl -X POST \
  -F "file=@input.wav" \
  -F "streaming=false" \
  http://localhost:6542/encode \
  -o tokens.npz

# Using client script
python encoder_client.py input.wav tokens.npz

# With streaming for long files
python encoder_client.py long_audio.wav tokens.npz --streaming
```

Parameters:
- `file`: Audio file (wav, mp3, flac, etc.)
- `streaming`: Process in chunks (default: false)

Returns: NPZ file with tokens

### POST /decode

Decode tokens to audio.

```bash
# Using curl
curl -X POST \
  -F "file=@tokens.npz" \
  -F "streaming=false" \
  http://localhost:6542/decode \
  -o output.wav

# Using client script
python decoder_client.py tokens.npz output.wav

# With streaming for long sequences
python decoder_client.py tokens.npz output.wav --streaming
```

Parameters:
- `file`: NPZ file with tokens
- `streaming`: Process in chunks (default: false)

Returns: WAV audio file (24kHz)

## Client Scripts

### Encoder Client

```bash
# Basic usage
python encoder_client.py input.wav tokens.npz

# Custom API URL
python encoder_client.py input.wav tokens.npz --api-url http://my-server:6542

# Streaming mode
python encoder_client.py long_audio.wav tokens.npz --streaming
```

### Decoder Client

```bash
# Basic usage
python decoder_client.py tokens.npz output.wav

# Custom API URL
python decoder_client.py tokens.npz output.wav --api-url http://my-server:6542

# Streaming mode
python decoder_client.py tokens.npz output.wav --streaming
```

## Docker Configuration

### Environment Variables

- `HUGGING_FACE_HUB_TOKEN`: HuggingFace API token (optional)
- `HF_TOKEN`: Alternative name for HuggingFace token
- `MIMI_MODEL_PATH`: Model path (default: `kyutai/mimi`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `6542`)

### Using a Local Model

1. Download the model:
   ```bash
   git lfs install
   git clone https://huggingface.co/kyutai/mimi
   ```

2. Mount it in docker-compose.yml:
   ```yaml
   volumes:
     - ./mimi:/app/mimi:ro
   ```

3. Set environment variable:
   ```yaml
   environment:
     - MIMI_MODEL_PATH=/app/mimi
   ```

### GPU Support

The docker-compose.yml includes GPU support via NVIDIA Container Toolkit.

**Prerequisites:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**CPU-only mode:**

Remove the `deploy` section from docker-compose.yml:
```yaml
# Remove or comment out:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

## Performance

### Encoding
- **GPU**: ~100-500× realtime
- **CPU**: ~10-50× realtime
- **Memory**: ~2GB VRAM + audio buffer

### Decoding
- **GPU**: ~100-500× realtime
- **CPU**: ~10-50× realtime
- **Memory**: ~2GB VRAM + audio buffer

### Recommendations

- Use **streaming mode** for files > 30 seconds
- Use **GPU** for production workloads
- Use **CPU** for development or low-throughput scenarios

## Monitoring

### Docker

```bash
# View logs
docker-compose logs -f

# Check container health
docker-compose ps

# Check resource usage
docker stats mimi-codec
```

### Metrics

The `/health` endpoint provides basic status. For production, consider adding:
- Prometheus metrics
- OpenTelemetry tracing
- Application performance monitoring (APM)

## Troubleshooting

### Server won't start

1. **Check GPU availability:**
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Check logs:**
   ```bash
   docker-compose logs mimi-codec-server
   ```

3. **Verify ports:**
   ```bash
   lsof -i :6542
   ```

### Model loading errors

1. **Check HuggingFace token:**
   ```bash
   echo $HUGGING_FACE_HUB_TOKEN
   ```

2. **Download model manually:**
   ```bash
   git lfs install
   git clone https://huggingface.co/kyutai/mimi
   # Then update docker-compose.yml to use local model
   ```

### Client connection errors

1. **Check server is running:**
   ```bash
   curl http://localhost:6542/health
   ```

2. **Check firewall:**
   ```bash
   sudo ufw status
   sudo ufw allow 6542/tcp
   ```

## Development

### Running tests

```bash
# Start the server
python server.py &

# Test with curl
curl http://localhost:6542/health

# Test encode
curl -X POST -F "file=@test.wav" http://localhost:6542/encode -o test.npz

# Test decode
curl -X POST -F "file=@test.npz" http://localhost:6542/decode -o test_reconstructed.wav
```

### Adding new endpoints

1. Edit `server.py`
2. Add the new endpoint function
3. Update this README
4. Rebuild the Docker image

## Production Deployment

### Security

1. **Use HTTPS**: Put behind a reverse proxy (nginx, traefik)
2. **Add authentication**: Implement API keys or OAuth
3. **Rate limiting**: Add rate limiting middleware
4. **Input validation**: Already included, but review for your use case

### Scaling

1. **Horizontal scaling**: Run multiple containers behind a load balancer
2. **Queue system**: Add Celery/RQ for asynchronous processing
3. **Caching**: Add Redis for token caching
4. **CDN**: Serve static assets via CDN

### Example with Traefik

```yaml
version: '3.8'

services:
  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock

  mimi-codec-server:
    build: .
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.mimi.rule=Host(`mimi.example.com`)"
      - "traefik.http.routers.mimi.entrypoints=websecure"
      - "traefik.http.routers.mimi.tls=true"
```

## License

See main project [LICENSE](LICENSE) file.

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Main README](README.md)

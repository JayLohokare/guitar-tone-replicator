#!/bin/bash
# Guitar Separation API v2 - Start Script

cd ~/guitar-api-v2

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run server on port 8766 (v2)
python3 server.py --port 8766
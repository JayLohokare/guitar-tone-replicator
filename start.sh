#!/bin/bash
# Tone Replicator - Start everything
# Usage: ./start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Start the Python API server
echo "🎸 Starting Tone Replicator API server..."
source venv/bin/activate
python server.py &
SERVER_PID=$!

echo "✅ API server running on http://localhost:8767 (PID: $SERVER_PID)"

# Start the Mac app
echo "🎸 Launching Tone Replicator app..."
if [ -f "ToneReplicatorApp/.build/release/ToneReplicatorApp" ]; then
    open ToneReplicatorApp/.build/release/ToneReplicatorApp
else
    echo "Building Mac app first..."
    cd ToneReplicatorApp && swift build -c release && cd ..
    open ToneReplicatorApp/.build/release/ToneReplicatorApp
fi

echo ""
echo "🎸 Tone Replicator is running!"
echo "   API: http://localhost:8767"
echo "   Press Ctrl+C to stop the server"

# Wait for server
wait $SERVER_PID
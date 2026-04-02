#!/bin/bash
# Guitar Separation API v2 - Cloudflare Tunnel Script

# Kill any existing tunnel
pkill -f "cloudflared tunnel" 2>/dev/null

# Start new tunnel to port 8766
cloudflared tunnel --url http://localhost:8766 > ~/guitar-api-v2/tunnel_output.log 2>&1 &

echo "Tunnel started. Check ~/guitar-api-v2/tunnel_output.log for URL"
#!/bin/bash

PORT=8000
# Find the first available port (default 8000)
while lsof -i :$PORT >/dev/null 2>&1; do
    ((PORT++))
done

python -m http.server $PORT &
SERVER_PID=$!

sleep 1
open "http://localhost:$PORT"

wait $SERVER_PID

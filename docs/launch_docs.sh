#!/bin/bash

PORT=8000
# Find the first available port (default 8000)
while lsof -i :$PORT >/dev/null 2>&1; do
    ((PORT++))
done

python -m http.server $PORT &
SERVER_PID=$!

# Open the docs in the default browser
sleep 1
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "http://localhost:$PORT"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "http://localhost:$PORT"
elif [[ "$OSTYPE" == "cygwin" ]]; then
    # Cygwin (typically for Windows)
    start "http://localhost:$PORT"
elif [[ "$OSTYPE" == "msys" ]]; then
    # Git Bash on Windows (MSYS)
    start "http://localhost:$PORT"
elif [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    start "http://localhost:$PORT"
else
    echo "Unknown OS, cannot open browser automatically."
fi

# Allow the server to run for some time (e.g., 30 seconds) before stopping
sleep 30

# Now gracefully terminate the server
kill $SERVER_PID
wait $SERVER_PID

#!/bin/bash

# Development script to run the full stack locally

set -e

# Function to kill processes on specified ports
kill_on_port() {
    PORT=$1
    echo "🔍 Checking for process on port $PORT..."
    PID=$(lsof -t -i:$PORT || true)
    
    if [ ! -z "$PID" ]; then
        echo "⚠️  Found process $PID on port $PORT. Terminating..."
        kill -9 $PID
        sleep 1 # Give it a moment to release the port
        echo "✅ Process terminated."
    else
        echo "✅ Port $PORT is free."
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    kill %1 %2 2>/dev/null || true
}
trap cleanup EXIT

echo "🚀 Starting Kent AI Agent in development mode..."

# Kill any existing processes on the ports we need
kill_on_port 8000
kill_on_port 3000

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating template..."
    cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key_here
BACKEND_URL=http://localhost:8000
EOF
    echo "📝 Please edit .env with your Google API key"
    exit 1
fi

# Start backend
echo "🐍 Starting Python backend..."
cd backend
if [ ! -d "venv" ]; then
    uv venv
    uv pip install -r requirements.txt
fi
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Start frontend
echo "⚛️  Starting Next.js frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Development servers started!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
wait
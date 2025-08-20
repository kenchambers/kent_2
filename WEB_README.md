# Kent AI Agent - Web Interface

A modern web interface for the Kent AI Agent featuring:

- 🎨 **Modern glassmorphism design** with gradients and blur effects
- 💻 **Terminal-style chat interface** mimicking macOS Terminal  
- 🤖 **Real-time AI responses** with conversation logging
- 📱 **Responsive design** works on desktop and mobile
- 🚀 **Deploy to Fly.io** with single command

## Architecture

```
User → Next.js Frontend → FastAPI Backend → Kent AI Agent → Google Gemini
```

- **Frontend**: Next.js with Tailwind CSS (React/JavaScript)
- **Backend**: FastAPI wrapping the existing Python agent
- **Database**: SQLite for conversation storage
- **Deployment**: Single container on Fly.io

## Quick Start

### Development

1. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your GOOGLE_API_KEY
   ```

2. **Run development servers**:
   ```bash
   ./dev.sh
   ```
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000

### Production Deployment

1. **Install Fly.io CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and deploy**:
   ```bash
   flyctl auth login
   ./deploy.sh
   ```

## Project Structure

```
kent_2/
├── frontend/           # Next.js React app
│   ├── app/
│   │   ├── page.js            # Main chat interface
│   │   ├── components/
│   │   │   └── Terminal.js    # Terminal chat component
│   │   └── api/chat/
│   │       └── route.js       # API proxy to backend
├── backend/            # FastAPI wrapper
│   ├── main.py                # FastAPI server + agent integration
│   └── requirements.txt       # Python dependencies
├── src/               # Existing Kent AI agent
├── Dockerfile         # Multi-stage build
├── fly.toml          # Fly.io configuration
├── deploy.sh         # Deployment script
└── dev.sh           # Development script
```

## Features

### Chat Interface
- **Mac Terminal styling** with traffic light buttons
- **Glassmorphism effects** with backdrop blur and transparency
- **Gradient backgrounds** from purple to blue to indigo
- **Real-time typing** with loading indicators
- **Session management** with unique session IDs
- **Message history** persistent across refreshes

### Backend API
- **FastAPI integration** with existing agent
- **SQLite storage** for conversation logging  
- **Session tracking** with timestamps
- **Error handling** with friendly error messages
- **Health checks** for monitoring

### Deployment
- **Single container** deployment to Fly.io
- **Persistent storage** for conversations and agent memory
- **Environment variables** for API keys
- **Automatic scaling** with auto-stop/start machines

## Environment Variables

```bash
GOOGLE_API_KEY=your_google_api_key_here
BACKEND_URL=http://localhost:8000  # For frontend dev
DATABASE_PATH=/app/data/conversations.db  # For production
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check  
- `POST /api/chat` - Main chat endpoint
- `GET /api/conversations/{session_id}` - Get conversation history
- `GET /api/sessions` - List all sessions

## Monitoring

```bash
# View logs
flyctl logs

# Check status  
flyctl status

# SSH into container
flyctl ssh console

# Scale machines
flyctl scale count 2
```

## Development Tips

- Backend runs on port 8000
- Frontend runs on port 3000 
- Database file created automatically
- Agent memory persists in vector_stores/
- Hot reload enabled for both frontend and backend

## Troubleshooting

**Backend not responding**: Check if Python dependencies are installed and GOOGLE_API_KEY is set

**Frontend can't reach backend**: Verify BACKEND_URL environment variable

**Agent memory issues**: Ensure vector_stores/ directory is properly mounted

**Deployment fails**: Check flyctl login status and app name conflicts
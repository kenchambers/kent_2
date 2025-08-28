"""
FastAPI backend that wraps the SelfImprovingAgent for web deployment.
"""
import asyncio
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_improving_agent.agent import SelfImprovingAgent
from self_improving_agent.utils import set_thinking_callback


class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"
    user_id: str = "default_user"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str


app = FastAPI(title="Kent AI Agent API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single agent instance - handles session isolation through fixes in the agent class
agent = None

db_path = os.getenv("DATABASE_PATH", "conversations.db")


async def get_agent():
    """Get or create the agent instance."""
    global agent
    if agent is None:
        agent = SelfImprovingAgent()
    return agent

def init_db():
    """Initialize SQLite database for conversation storage."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            agent_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def save_conversation(session_id: str, user_message: str, agent_response: str):
    """Save conversation to database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Update or create session
    cursor.execute("""
        INSERT OR REPLACE INTO sessions (id, last_active)
        VALUES (?, CURRENT_TIMESTAMP)
    """, (session_id,))
    
    # Save conversation
    cursor.execute("""
        INSERT INTO conversations (session_id, user_message, agent_response)
        VALUES (?, ?, ?)
    """, (session_id, user_message, agent_response))
    
    conn.commit()
    conn.close()


@app.on_event("startup")
async def startup_event():
    """Initialize database and agent on startup."""
    init_db()
    await get_agent()

@app.on_event("shutdown")
async def shutdown_event():
    """Close agent resources on shutdown."""
    if agent:
        await agent.aclose()
        print("Agent connection closed.")


# Mount static files for the Next.js frontend
frontend_static_path = os.path.join(os.path.dirname(__file__), "..", "frontend", ".next", "static")
frontend_public_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")

if os.path.exists(frontend_static_path):
    app.mount("/_next/static", StaticFiles(directory=frontend_static_path), name="nextstatic")

if os.path.exists(frontend_public_path):
    app.mount("/static", StaticFiles(directory=frontend_public_path), name="public")


# Mount the entire Next.js static build at root (this will serve your React app)
frontend_build_static = os.path.join(os.path.dirname(__file__), "..", "frontend", ".next", "static")
if os.path.exists(frontend_build_static):
    # Serve Next.js static files at /_next/static
    app.mount("/_next", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend", ".next")), name="next")


@app.get("/")
async def root():
    """Root endpoint - serve the EXACT same frontend as dev.sh"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kent AI Agent Terminal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <main class="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
        <div class="container mx-auto max-w-6xl h-screen flex items-center justify-center">
            <div class="backdrop-blur-xl bg-white/10 rounded-2xl border border-white/20 shadow-2xl overflow-hidden w-full max-h-[calc(100vh-2rem)]">
                <div id="terminal-root"></div>
            </div>
        </div>
    </main>
    
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    
    <script type="text/babel">
        const { useState, useRef, useEffect } = React;

        function Terminal() {
          const [messages, setMessages] = useState([
            { type: 'system', content: 'Kent AI Agent initialized. Type your message below.' }
          ]);
          const [input, setInput] = useState('');
          const [isLoading, setIsLoading] = useState(false);
          const [isStreaming, setIsStreaming] = useState(false);
          const [thinkingSteps, setThinkingSteps] = useState([]);
          const [showThinking, setShowThinking] = useState(false);
          const [sessionId] = useState(() => `session_${ Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
          const messagesEndRef = useRef(null);
          const inputRef = useRef(null);

          const scrollToBottom = () => {
            messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
          };

          useEffect(() => {
            scrollToBottom();
          }, [messages, thinkingSteps]);

          useEffect(() => {
            inputRef.current?.focus();
          }, []);

          const sendMessage = async (e) => {
            e.preventDefault();
            if (!input.trim() || isLoading || isStreaming) return;

            const userMessage = input.trim();
            setInput('');
            setIsLoading(true);
            setIsStreaming(true);
            setThinkingSteps([]);
            setShowThinking(false);

            setMessages(prev => [...prev, { type: 'user', content: userMessage }]);

            try {
              const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  message: userMessage,
                  session_id: sessionId
                })
              });

              if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
              }

              const reader = response.body?.getReader();
              const decoder = new TextDecoder();

              if (!reader) {
                throw new Error('No response body reader available');
              }

              let buffer = '';
              
              while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\\n');
                
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                  if (line.startsWith('data: ')) {
                    try {
                      const data = JSON.parse(line.slice(6));
                      
                      if (data.type === 'thinking') {
                        setThinkingSteps(prev => [...prev, {
                          id: Date.now() + Math.random(),
                          content: data.content,
                          timestamp: data.timestamp || new Date().toISOString(),
                          isNew: true
                        }]);
                        setShowThinking(true);
                      } else if (data.type === 'response') {
                        setThinkingSteps(prev => prev.map(step => ({...step, isNew: false, fading: true})));
                        setTimeout(() => {
                          setThinkingSteps([]);
                          setShowThinking(false);
                          setMessages(prev => [...prev, { 
                            type: 'agent', 
                            content: data.content,
                            timestamp: data.timestamp 
                          }]);
                        }, 1500);
                      } else if (data.type === 'complete') {
                        setIsStreaming(false);
                        setIsLoading(false);
                      } else if (data.type === 'error') {
                        setThinkingSteps([]);
                        setShowThinking(false);
                        setMessages(prev => [...prev, { 
                          type: 'error', 
                          content: data.content
                        }]);
                        setIsStreaming(false);
                        setIsLoading(false);
                      }
                    } catch (parseError) {
                      console.error('Error parsing SSE data:', parseError);
                    }
                  }
                }
              }
              
            } catch (error) {
              console.error('Error sending message:', error);
              setThinkingSteps([]);
              setShowThinking(false);
              setMessages(prev => [...prev, { 
                type: 'error', 
                content: `Error: ${error.message}. Make sure the backend server is running.`
              }]);
            } finally {
              setIsLoading(false);
              setIsStreaming(false);
              setThinkingSteps([]);
              setShowThinking(false);
            }
          };

          const formatMessage = (message) => {
            return message.split('\\n').map((line, index) => {
              let className = "";
              let content = line;
              
              if (line.includes('thinking') || line.includes('analyzing')) {
                className = "text-yellow-400 italic";
              } else if (line.includes('searching') || line.includes('loading')) {
                className = "text-cyan-400";
              } else if (line.includes('error') || line.includes('Error')) {
                className = "text-red-400";
              } else if (line.includes('memory') || line.includes('episodic')) {
                className = "text-green-400";
              } else {
                className = "text-gray-200";
              }
              
              return (
                <div key={index} className={className}>
                  {content || <br />}
                </div>
              );
            });
          };

          return (
            <div className="h-[calc(100vh-4rem)] max-h-[800px] flex flex-col">
              <div className="bg-gray-800/50 border-b border-white/10 p-4 flex items-center space-x-2">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                </div>
                <div className="text-gray-300 text-sm ml-4">Kent AI Terminal</div>
                <div className="ml-auto text-xs text-gray-400">{sessionId.slice(0, 12)}...</div>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-sm">
                {messages.map((message, index) => (
                  <div key={index} className="group">
                    {message.type === 'system' && (
                      <div className="text-green-400 opacity-80">
                        <span className="text-gray-500">system:</span> {message.content}
                      </div>
                    )}
                    
                    {message.type === 'user' && (
                      <div className="text-blue-300">
                        <span className="text-gray-400">you@kent:</span>
                        <span className="text-purple-300">~</span>
                        <span className="text-gray-300">$</span> {message.content}
                      </div>
                    )}
                    
                    {message.type === 'agent' && (
                      <div className="text-gray-200 pl-4 border-l-2 border-purple-500/30">
                        <div className="text-xs text-gray-500 mb-1">
                          {message.timestamp && new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                        <div className="whitespace-pre-wrap">{formatMessage(message.content)}</div>
                      </div>
                    )}
                    
                    {message.type === 'error' && (
                      <div className="text-red-400 pl-4 border-l-2 border-red-500/50">
                        {message.content}
                      </div>
                    )}
                  </div>
                ))}

                {(isStreaming || showThinking) && thinkingSteps.length > 0 && (
                  <div className="space-y-2 mb-4">
                    <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider flex items-center space-x-2">
                      <div className="animate-spin w-3 h-3 border border-purple-500 border-t-transparent rounded-full"></div>
                      <span>Thinking Process</span>
                    </div>
                    {thinkingSteps.map((step, index) => (
                      <div 
                        key={step.id} 
                        className={`
                          text-gray-400 pl-4 border-l-2 border-purple-500/30 bg-purple-500/5 rounded-r-lg p-2
                          transition-all duration-1000 ease-out
                          ${step.fading ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}
                          ${step.isNew ? 'animate-pulse' : ''}
                        `}
                        style={{
                          animationDelay: `${index * 0.1}s`,
                          transform: step.fading ? 'translateX(-10px)' : 'translateX(0)'
                        }}
                      >
                        <div className="flex items-start space-x-2">
                          <div className="animate-spin w-3 h-3 border border-purple-500 border-t-transparent rounded-full flex-shrink-0 mt-0.5"></div>
                          <div className="font-mono text-xs leading-relaxed break-words flex-1">
                            {step.content}
                          </div>
                        </div>
                        {step.timestamp && (
                          <div className="text-xs opacity-50 mt-1 ml-5">
                            {new Date(step.timestamp).toLocaleTimeString()}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {isLoading && thinkingSteps.length === 0 && (
                  <div className="text-gray-400 pl-4 border-l-2 border-purple-500/30">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full"></div>
                      <span>Connecting to Kent...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              <div className="border-t border-white/10 p-4">
                <form onSubmit={sendMessage} className="flex items-center space-x-2">
                  <div className="text-blue-300 text-sm font-mono">
                    <span className="text-gray-400">you@kent:</span>
                    <span className="text-purple-300">~</span>
                    <span className="text-gray-300">$</span>
                  </div>
                  <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={isStreaming ? "Kent is processing..." : "Type your message..."}
                    disabled={isLoading || isStreaming}
                    className="flex-1 bg-transparent border-none outline-none text-gray-200 font-mono text-sm placeholder-gray-500 disabled:opacity-50"
                  />
                  <button
                    type="submit"
                    disabled={!input.trim() || isLoading || isStreaming}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm transition-colors"
                  >
                    {isStreaming ? 'Processing...' : 'Send'}
                  </button>
                </form>
              </div>
            </div>
          );
        }

        ReactDOM.render(<Terminal />, document.getElementById('terminal-root'));
    </script>
</body>
</html>
    """)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/chat/stream")
async def chat_stream(message: ChatMessage):
    """Streaming chat endpoint with both native streaming and detailed thinking logs."""
    thinking_queue = asyncio.Queue()
    
    # Get the current event loop at the start of the request
    current_loop = asyncio.get_running_loop()
    
    def thinking_callback(thinking_message: str):
        """Callback to capture detailed thinking logs from agent."""
        try:
            # Use the captured loop reference to safely add to queue from any thread
            current_loop.call_soon_threadsafe(thinking_queue.put_nowait, thinking_message)
        except Exception as e:
            print(f"Error in thinking callback: {e}")
    
    async def generate_response():
        try:
            agent_instance = await get_agent()
            
            # Set up the thinking callback to capture detailed agent logs
            set_thinking_callback(thinking_callback)
            
            # Start agent processing using native streaming
            agent_stream = agent_instance.astream(
                message.message,
                session_id=message.session_id,
                user_id=message.user_id,
            )
            agent_task = asyncio.create_task(anext(agent_stream))
            
            response_content = None
            
            # Stream both detailed thinking and high-level progress
            while True:
                done_tasks = []
                
                # Check for detailed thinking messages
                try:
                    thinking_msg = thinking_queue.get_nowait()
                    yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_msg, 'timestamp': datetime.now().isoformat()})}\n\n"
                except asyncio.QueueEmpty:
                    pass
                
                # Check for high-level progress from native streaming
                if agent_task.done():
                    try:
                        chunk = await agent_task
                        
                        if chunk["type"] == "response":
                            response_content = chunk["content"]
                            
                        # Send the high-level progress
                        chunk_data = {
                            "type": chunk["type"],
                            "content": chunk["content"],
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        if chunk["type"] == "response":
                            break
                        
                        # Get next chunk
                        agent_task = asyncio.create_task(anext(agent_stream))
                        
                    except StopAsyncIteration:
                        # Agent streaming is complete
                        break
                    except Exception as e:
                        print(f"Error in agent streaming: {e}")
                        break
                
                # Small delay to avoid busy waiting
                await asyncio.sleep(0.05)
            
            # Clear the callback
            set_thinking_callback(None)
            
            # Send any remaining thinking messages
            while not thinking_queue.empty():
                try:
                    thinking_msg = thinking_queue.get_nowait()
                    yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_msg, 'timestamp': datetime.now().isoformat()})}\n\n"
                except asyncio.QueueEmpty:
                    break
            
            # Save conversation if we got a response
            if response_content:
                save_conversation(message.session_id, message.message, response_content)
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'session_id': message.session_id, 'timestamp': datetime.now().isoformat()})}\n\n"
            
        except Exception as e:
            # Clear the callback on error
            set_thinking_callback(None)
            yield f"data: {json.dumps({'type': 'error', 'content': f'Agent error: {str(e)}', 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/plain")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Non-streaming chat endpoint (fallback)."""
    try:
        agent_instance = await get_agent()
        
        # Get response from agent
        response = await agent_instance.arun(
            message.message, session_id=message.session_id, user_id=message.user_id
        )
        
        # Save conversation
        save_conversation(message.session_id, message.message, response)
        
        return ChatResponse(
            response=response,
            session_id=message.session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/api/conversations/{session_id}")
async def get_conversations(session_id: str):
    """Get conversation history for a session."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_message, agent_response, timestamp
        FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    
    conversations = []
    for row in cursor.fetchall():
        conversations.extend([
            {"type": "user", "message": row[0], "timestamp": row[2]},
            {"type": "agent", "message": row[1], "timestamp": row[2]}
        ])
    
    conn.close()
    return {"conversations": conversations}


@app.get("/api/sessions")
async def get_sessions():
    """Get all session IDs."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, created_at, last_active
        FROM sessions
        ORDER BY last_active DESC
    """)
    
    sessions = [{"id": row[0], "created_at": row[1], "last_active": row[2]} 
               for row in cursor.fetchall()]
    
    conn.close()
    return {"sessions": sessions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
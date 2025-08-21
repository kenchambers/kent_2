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

# Initialize the agent
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

# Mount static files for frontend (if available)
frontend_static_path = os.path.join(os.path.dirname(__file__), "..", "frontend", ".next", "static")
frontend_public_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")

if os.path.exists(frontend_static_path):
    app.mount("/_next/static", StaticFiles(directory=frontend_static_path), name="nextstatic")

if os.path.exists(frontend_public_path):
    app.mount("/public", StaticFiles(directory=frontend_public_path), name="public")


@app.get("/")
async def root():
    """Root endpoint - serve React terminal interface."""
    # Create a complete React app that works with our streaming API
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kent AI Agent Terminal</title>
        <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', monospace;
                margin: 0;
                padding: 0;
            }}
            
            @keyframes fadeInSlide {{
                from {{
                    opacity: 0;
                    transform: translateX(20px) translateY(-10px);
                }}
                to {{
                    opacity: 1;
                    transform: translateX(0) translateY(0);
                }}
            }}
            
            @keyframes thinking-glow {{
                0%, 100% {{ box-shadow: 0 0 5px rgba(168, 85, 247, 0.4); }}
                50% {{ box-shadow: 0 0 15px rgba(168, 85, 247, 0.6), 0 0 25px rgba(168, 85, 247, 0.3); }}
            }}
            
            .thinking-container {{
                animation: thinking-glow 2s ease-in-out infinite;
            }}
            
            /* Custom scrollbar for terminal */
            .terminal-scrollbar {{
                scroll-behavior: smooth;
            }}
            .terminal-scrollbar::-webkit-scrollbar {{
                width: 8px;
            }}
            .terminal-scrollbar::-webkit-scrollbar-track {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }}
            .terminal-scrollbar::-webkit-scrollbar-thumb {{
                background: rgba(168, 85, 247, 0.6);
                border-radius: 4px;
            }}
            .terminal-scrollbar::-webkit-scrollbar-thumb:hover {{
                background: rgba(168, 85, 247, 0.8);
            }}
        </style>
    </head>
    <body class="min-h-screen p-4">
        <div id="root"></div>
        
        <script type="text/babel">
            const {{ useState, useRef, useEffect }} = React;
            
            function TerminalApp() {{
                const [messages, setMessages] = useState([
                    {{ type: 'system', content: '🚀 Kent AI Agent Terminal v2.0 - Streaming enabled!' }},
                    {{ type: 'info', content: '📚 Externalized Long-Term Memory\\n\\nInstead of keeping conversations in a growing text file that gets passed to the model, Kent stores long-term knowledge and conversation history in FAISS vector stores.\\n\\n🔍 What it is: A vector store is a database optimized for semantic search. It does not find keywords; it finds concepts and meanings.\\n\\n🧠 How it helps: The vast majority of Kent\\'s memory lives outside the context window in these vector stores. This allows the memory to grow to a virtually unlimited size without ever overwhelming the LLM.\\n\\n💡 Key Benefits:\\n• Semantic understanding over keyword matching\\n• Unlimited memory capacity\\n• Efficient context retrieval\\n• Persistent learning across sessions\\n\\nTry asking Kent about something from a previous conversation, or explore complex topics that build on each other!' }}
                ]);
                const [input, setInput] = useState('');
                const [isStreaming, setIsStreaming] = useState(false);
                const [thinkingSteps, setThinkingSteps] = useState([]);
                const [sessionId] = useState(() => `session_${{Date.now()}}_${{Math.random().toString(36).substr(2, 9)}}`);
                const messagesEndRef = useRef(null);
                const inputRef = useRef(null);
                
                useEffect(() => {{
                    // Scroll to bottom when new messages or thinking steps are added
                    if (messagesEndRef.current) {{
                        messagesEndRef.current.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
                    }}
                }}, [messages, thinkingSteps]);
                
                useEffect(() => {{
                    // Also scroll to bottom on initial load
                    if (messagesEndRef.current) {{
                        messagesEndRef.current.scrollIntoView({{ behavior: 'smooth' }});
                    }}
                }}, []);
                
                // Function to get icon and color for thinking step types
                const getThinkingStepStyle = (content) => {{
                    if (content.includes('Initial Analysis') || content.includes('Routing')) return {{ icon: '🧭', color: 'text-blue-400', bgColor: 'bg-blue-500/10', borderColor: 'border-blue-500/30', title: 'Analysis & Routing' }};
                    if (content.includes('Emotional Context')) return {{ icon: '💭', color: 'text-purple-400', bgColor: 'bg-purple-500/10', borderColor: 'border-purple-500/30', title: 'Emotional Context' }};
                    if (content.includes('Memory') || content.includes('Checking') || content.includes('Retrieved')) return {{ icon: '🧠', color: 'text-green-400', bgColor: 'bg-green-500/10', borderColor: 'border-green-500/30', title: 'Memory Operations' }};
                    if (content.includes('Conscience')) return {{ icon: '⚖️', color: 'text-yellow-400', bgColor: 'bg-yellow-500/10', borderColor: 'border-yellow-500/30', title: 'Self-Correction' }};
                    if (content.includes('Core Identity') || content.includes('Belief')) return {{ icon: '🆔', color: 'text-indigo-400', bgColor: 'bg-indigo-500/10', borderColor: 'border-indigo-500/30', title: 'Core Identity' }};
                    if (content.includes('Generating Response')) return {{ icon: '✨', color: 'text-pink-400', bgColor: 'bg-pink-500/10', borderColor: 'border-pink-500/30', title: 'Response Generation' }};
                    if (content.includes('Summary')) return {{ icon: '📝', color: 'text-cyan-400', bgColor: 'bg-cyan-500/10', borderColor: 'border-cyan-500/30', title: 'Summary Update' }};
                    if (content.includes('Inner Monologue')) return {{ icon: '🤔', color: 'text-orange-400', bgColor: 'bg-orange-500/10', borderColor: 'border-orange-500/30', title: 'Internal Reasoning' }};
                    if (content.includes('Parallel')) return {{ icon: '⚡', color: 'text-teal-400', bgColor: 'bg-teal-500/10', borderColor: 'border-teal-500/30', title: 'Parallel Processing' }};
                    return {{ icon: '🔍', color: 'text-gray-400', bgColor: 'bg-gray-500/10', borderColor: 'border-gray-500/30', title: 'Processing' }};
                }};

                // Function to format thinking content for better display
                const formatThinkingContent = (content) => {{
                    // Remove the header markers (--- text ---)
                    let formatted = content.replace(/^--- (.+) ---\\s*$/gm, '');
                    
                    // Try to parse JSON for inner monologue and other structured content
                    try {{
                        if (formatted.trim().startsWith('{{') && formatted.trim().endsWith('}}')) {{
                            const parsed = JSON.parse(formatted.trim());
                            return JSON.stringify(parsed, null, 2);
                        }}
                    }} catch (e) {{
                        // Not JSON, continue with regular formatting
                    }}
                    
                    // Clean up extra whitespace
                    formatted = formatted.trim();
                    
                    // If it's still mostly empty after removing headers, show the original
                    if (formatted.length < 10) {{
                        return content;
                    }}
                    
                    return formatted;
                }};
                
                useEffect(() => {{
                    inputRef.current?.focus();
                }}, []);
                
                const sendMessage = async (e) => {{
                    e.preventDefault();
                    if (!input.trim() || isStreaming) return;
                    
                    const userMessage = input.trim();
                    setInput('');
                    setIsStreaming(true);
                    setThinkingSteps([]);
                    
                    setMessages(prev => [...prev, {{ type: 'user', content: userMessage }}]);
                    
                    try {{
                        const response = await fetch('/api/chat/stream', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ message: userMessage, session_id: sessionId }})
                        }});
                        
                        if (!response.ok) throw new Error(`HTTP error! status: ${{response.status}}`);
                        
                        const reader = response.body?.getReader();
                        const decoder = new TextDecoder();
                        let buffer = '';
                        
                        while (true) {{
                            const {{ done, value }} = await reader.read();
                            if (done) break;
                            
                            buffer += decoder.decode(value, {{ stream: true }});
                            const lines = buffer.split('\\n');
                            buffer = lines.pop() || '';
                            
                            for (const line of lines) {{
                                if (line.startsWith('data: ')) {{
                                    try {{
                                        const data = JSON.parse(line.slice(6));
                                        if (data.type === 'thinking') {{
                                            setThinkingSteps(prev => [...prev, {{ 
                                                id: Date.now() + Math.random(), 
                                                content: data.content, 
                                                timestamp: data.timestamp,
                                                isNew: true
                                            }}]);
                                        }} else if (data.type === 'response') {{
                                            setThinkingSteps([]);
                                            setMessages(prev => [...prev, {{ 
                                                type: 'agent', 
                                                content: data.content,
                                                timestamp: data.timestamp 
                                            }}]);
                                        }} else if (data.type === 'complete') {{
                                            setIsStreaming(false);
                                        }} else if (data.type === 'error') {{
                                            setThinkingSteps([]);
                                            setMessages(prev => [...prev, {{ type: 'error', content: data.content }}]);
                                            setIsStreaming(false);
                                        }}
                                    }} catch (e) {{ console.error('Parse error:', e); }}
                                }}
                            }}
                        }}
                    }} catch (error) {{
                        console.error('Error:', error);
                        setThinkingSteps([]);
                        setMessages(prev => [...prev, {{ type: 'error', content: `Error: ${{error.message}}` }}]);
                        setIsStreaming(false);
                    }}
                }};
                
                return (
                    <div className="container mx-auto max-w-6xl">
                        <div className="text-center mb-8 pt-8">
                            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
                                Kent AI Agent
                            </h1>
                            <p className="text-gray-300 text-lg">Interactive streaming terminal</p>
                        </div>
                        
                        <div className="backdrop-blur-xl bg-white/10 rounded-2xl border border-white/20 shadow-2xl overflow-hidden">
                            <div className="bg-gray-800/50 border-b border-white/10 p-4 flex items-center space-x-2">
                                <div className="flex space-x-2">
                                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                </div>
                                <div className="text-gray-300 text-sm ml-4">Kent AI Terminal</div>
                                <div className="ml-auto text-xs text-gray-400">{{sessionId.slice(0, 12)}}...</div>
                            </div>
                            
                            <div className="h-[500px] overflow-y-auto p-4 space-y-4 font-mono text-sm terminal-scrollbar">
                                {{messages.map((message, index) => (
                                    <div key={{index}} className="group">
                                        {{message.type === 'system' && (
                                            <div className="text-green-400 opacity-80">
                                                <span className="text-gray-500">system:</span> {{message.content}}
                                            </div>
                                        )}}
                                        
                                        {{message.type === 'info' && (
                                            <div className="text-cyan-300 pl-4 border-l-4 border-cyan-500/50 bg-cyan-500/10 rounded-r-lg py-3 pr-3 mb-2">
                                                <div className="text-xs text-cyan-400 mb-2 font-semibold uppercase tracking-wider">
                                                    📚 System Information
                                                </div>
                                                <div className="whitespace-pre-wrap text-sm leading-relaxed">{{message.content}}</div>
                                            </div>
                                        )}}
                                        
                                        {{message.type === 'user' && (
                                            <div className="text-blue-300">
                                                <span className="text-gray-400">you@kent:</span>
                                                <span className="text-purple-300">~</span>
                                                <span className="text-gray-300">$ </span>{{message.content}}
                                            </div>
                                        )}}
                                        
                                        {{message.type === 'agent' && (
                                            <div className="text-gray-200 pl-4 border-l-2 border-purple-500/30">
                                                <div className="text-xs text-gray-500 mb-1">
                                                    {{message.timestamp && new Date(message.timestamp).toLocaleTimeString()}}
                                                </div>
                                                <div className="whitespace-pre-wrap">{{message.content}}</div>
                                            </div>
                                        )}}
                                        
                                        {{message.type === 'error' && (
                                            <div className="text-red-400 pl-4 border-l-2 border-red-500/50">
                                                {{message.content}}
                                            </div>
                                        )}}
                                    </div>
                                ))}}
                                
                                {{isStreaming && thinkingSteps.length > 0 && (
                                    <div className="space-y-2 mb-4 thinking-container rounded-lg p-3 border border-purple-500/20">
                                        <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider flex items-center space-x-2">
                                            <div className="animate-spin w-3 h-3 border border-purple-500 border-t-transparent rounded-full"></div>
                                            <span>Kent's Cognitive Process</span>
                                        </div>
                                        {{thinkingSteps.map((step, index) => {{
                                            const style = getThinkingStepStyle(step.content);
                                            return (
                                                <div 
                                                    key={{step.id}} 
                                                    className={{`
                                                        ${{style.bgColor}} ${{style.borderColor}} ${{style.color}}
                                                        border-l-4 pl-4 pr-3 py-2 rounded-r-lg backdrop-blur-sm
                                                        transform transition-all duration-500 ease-out
                                                        ${{step.isNew ? 'animate-pulse' : ''}}
                                                        opacity-0 translate-x-4
                                                    `}}
                                                    style={{{{
                                                        animation: `fadeInSlide 0.6s ease-out ${{index * 0.1}}s forwards`
                                                    }}}}
                                                >
                                                    <div className="flex items-start space-x-2">
                                                        <span className="text-base flex-shrink-0 mt-0.5">{{style.icon}}</span>
                                                        <div className="flex-1 min-w-0">
                                                            <div className="font-semibold text-sm mb-2 opacity-90">
                                                                {{style.title}}
                                                            </div>
                                                            <div className="font-mono text-xs leading-relaxed break-words">
                                                                {{(() => {{
                                                                    const formattedContent = formatThinkingContent(step.content);
                                                                    const isLongContent = formattedContent.length > 200;
                                                                    
                                                                    return isLongContent ? (
                                                                        <details className="group">
                                                                            <summary className="cursor-pointer hover:text-white transition-colors list-none flex items-center space-x-2">
                                                                                <span>{{formattedContent.substring(0, 200).replace(/\\n/g, ' ')}}...</span>
                                                                                <span className="ml-2 text-xs opacity-60 group-open:hidden bg-current/20 px-2 py-1 rounded">[expand]</span>
                                                                            </summary>
                                                                            <div className="mt-3 pl-3 border-l-2 border-current/20 bg-black/20 rounded p-3">
                                                                                <pre className="whitespace-pre-wrap text-xs">{{formattedContent}}</pre>
                                                                            </div>
                                                                        </details>
                                                                    ) : (
                                                                        <pre className="whitespace-pre-wrap">{{formattedContent}}</pre>
                                                                    );
                                                                }})()}}
                                                            </div>
                                                            {{step.timestamp && (
                                                                <div className="text-xs opacity-50 mt-2 flex items-center space-x-1">
                                                                    <span>⏱</span>
                                                                    <span>{{new Date(step.timestamp).toLocaleTimeString()}}</span>
                                                                </div>
                                                            )}}
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        }})}}
                                    </div>
                                )}}
                                
                                <div ref={{messagesEndRef}} />
                            </div>
                            
                            <div className="border-t border-white/10 p-4">
                                <form onSubmit={{sendMessage}} className="flex items-center space-x-2">
                                    <div className="text-blue-300 text-sm font-mono">
                                        <span className="text-gray-400">you@kent:</span>
                                        <span className="text-purple-300">~</span>
                                        <span className="text-gray-300">$ </span>
                                    </div>
                                    <input
                                        ref={{inputRef}}
                                        type="text"
                                        value={{input}}
                                        onChange={{(e) => setInput(e.target.value)}}
                                        placeholder={{isStreaming ? "Kent is processing..." : "Type your message..."}}
                                        disabled={{isStreaming}}
                                        className="flex-1 bg-transparent border-none outline-none text-gray-200 font-mono text-sm placeholder-gray-500 disabled:opacity-50"
                                    />
                                    <button
                                        type="submit"
                                        disabled={{!input.trim() || isStreaming}}
                                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm transition-colors"
                                    >
                                        {{isStreaming ? 'Processing...' : 'Send'}}
                                    </button>
                                </form>
                            </div>
                        </div>
                        
                        <div className="text-center mt-4">
                            <p className="text-gray-400 text-xs">
                                Powered by LangGraph + Google Gemini 2.5 Pro • Streaming enabled • Session: {{sessionId.slice(0, 8)}}
                            </p>
                        </div>
                    </div>
                );
            }}
            
            ReactDOM.render(<TerminalApp />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/chat/stream")
async def chat_stream(message: ChatMessage):
    """Streaming chat endpoint with real thinking process."""
    thinking_queue = asyncio.Queue()
    
    def thinking_callback(thinking_message: str):
        """Callback to capture thinking logs from agent."""
        try:
            # Use asyncio.create_task to put item in queue from sync context
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(thinking_queue.put_nowait, thinking_message)
        except Exception as e:
            print(f"Error in thinking callback: {e}")
    
    async def generate_response():
        try:
            agent_instance = await get_agent()
            
            # Set up the thinking callback to capture agent logs
            set_thinking_callback(thinking_callback)
            
            # Start agent processing in a separate task
            agent_task = asyncio.create_task(agent_instance.arun(message.message, session_id=message.session_id))
            
            # Stream thinking messages as they come in
            response = None
            while True:
                try:
                    # Check if we have thinking messages to send
                    try:
                        thinking_msg = thinking_queue.get_nowait()
                        yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_msg, 'timestamp': datetime.now().isoformat()})}\n\n"
                    except asyncio.QueueEmpty:
                        pass
                    
                    # Check if agent task is complete
                    if agent_task.done():
                        response = await agent_task
                        break
                    
                    # Small delay to avoid busy waiting
                    await asyncio.sleep(0.1)
                
                except Exception as e:
                    print(f"Error in streaming loop: {e}")
                    break
            
            # Clear the callback
            set_thinking_callback(None)
            
            # Send any remaining thinking messages
            while not thinking_queue.empty():
                try:
                    thinking_msg = thinking_queue.get_nowait()
                    yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_msg, 'timestamp': datetime.now().isoformat()})}\n\n"
                except asyncio.QueueEmpty:
                    break
            
            if response:
                # Send final response
                yield f"data: {json.dumps({'type': 'response', 'content': response, 'timestamp': datetime.now().isoformat()})}\n\n"
                
                # Save conversation
                save_conversation(message.session_id, message.message, response)
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'session_id': message.session_id})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No response from agent', 'timestamp': datetime.now().isoformat()})}\n\n"
            
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
        response = await agent_instance.arun(message.message, session_id=message.session_id)
        
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
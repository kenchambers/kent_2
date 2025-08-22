'use client'

import { useState, useRef, useEffect } from 'react'

export function Terminal() {
  const [messages, setMessages] = useState([
    { type: 'system', content: 'Kent AI Agent initialized. Type your message below.' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [thinkingSteps, setThinkingSteps] = useState([])
  const [showThinking, setShowThinking] = useState(false)
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, thinkingSteps])

  useEffect(() => {
    // Focus input on mount
    inputRef.current?.focus()
  }, [])

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading || isStreaming) return

    const userMessage = input.trim()
    setInput('')
    setIsLoading(true)
    setIsStreaming(true)
    setThinkingSteps([])
    setShowThinking(false)

    // Add user message
    setMessages(prev => [...prev, { type: 'user', content: userMessage }])

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
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body reader available')
      }

      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || ''
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.type === 'thinking') {
                setThinkingSteps(prev => [...prev, {
                  id: Date.now() + Math.random(),
                  content: data.content,
                  timestamp: data.timestamp || new Date().toISOString(),
                  isNew: true
                }])
                setShowThinking(true)
              } else if (data.type === 'response') {
                // Start fade out animation for thinking steps first
                setThinkingSteps(prev => prev.map(step => ({...step, isNew: false, fading: true})))
                // Clear thinking steps after fade animation
                setTimeout(() => {
                  setThinkingSteps([])
                  setShowThinking(false)
                  // Add response after thinking is cleared
                  setMessages(prev => [...prev, { 
                    type: 'agent', 
                    content: data.content,
                    timestamp: data.timestamp 
                  }])
                }, 1500)
              } else if (data.type === 'complete') {
                // Stream completed
                setIsStreaming(false)
                setIsLoading(false)
              } else if (data.type === 'error') {
                setThinkingSteps([])
                setShowThinking(false)
                setMessages(prev => [...prev, { 
                  type: 'error', 
                  content: data.content
                }])
                setIsStreaming(false)
                setIsLoading(false)
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError)
            }
          }
        }
      }
      
    } catch (error) {
      console.error('Error sending message:', error)
      setThinkingSteps([])
      setShowThinking(false)
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: `Error: ${error.message}. Make sure the backend server is running.`
      }])
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
      setThinkingSteps([])
      setShowThinking(false)
    }
  }

  const formatMessage = (message) => {
    // Enhanced formatting for terminal-like display
    return message.split('\n').map((line, index) => {
      // Color code different types of content
      let className = ""
      let content = line
      
      // Detect different types of agent output
      if (line.includes('thinking') || line.includes('analyzing')) {
        className = "text-yellow-400 italic"
      } else if (line.includes('searching') || line.includes('loading')) {
        className = "text-cyan-400"
      } else if (line.includes('error') || line.includes('Error')) {
        className = "text-red-400"
      } else if (line.includes('memory') || line.includes('episodic')) {
        className = "text-green-400"
      } else {
        className = "text-gray-200"
      }
      
      return (
        <div key={index} className={className}>
          {content || <br />}
        </div>
      )
    })
  }

  return (
    <div className="h-[calc(100vh-4rem)] max-h-[800px] flex flex-col">
      {/* Terminal Header */}
      <div className="bg-gray-800/50 border-b border-white/10 p-4 flex items-center space-x-2">
        <div className="flex space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
        </div>
        <div className="text-gray-300 text-sm ml-4">Kent AI Terminal</div>
        <div className="ml-auto text-xs text-gray-400">{sessionId.slice(0, 12)}...</div>
      </div>

      {/* Messages Area */}
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

      {/* Input Area */}
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
  )
}
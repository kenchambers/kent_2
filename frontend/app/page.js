'use client'

import { Terminal } from './components/Terminal'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-8 pt-8">
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
            Kent AI Agent
          </h1>
          <p className="text-gray-300 text-lg">
            Interactive terminal interface powered by self-improving AI
          </p>
        </div>
        
        <div className="backdrop-blur-xl bg-white/10 rounded-2xl border border-white/20 shadow-2xl overflow-hidden">
          <Terminal />
        </div>
      </div>
    </main>
  )
}

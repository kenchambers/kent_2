'use client'

import { Terminal } from './components/Terminal'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
      <div className="container mx-auto max-w-6xl h-screen flex items-center justify-center">
        <div className="backdrop-blur-xl bg-white/10 rounded-2xl border border-white/20 shadow-2xl overflow-hidden w-full max-h-[calc(100vh-2rem)]">
          <Terminal />
        </div>
      </div>
    </main>
  )
}

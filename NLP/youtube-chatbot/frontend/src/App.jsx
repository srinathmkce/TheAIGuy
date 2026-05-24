import { useState } from 'react'

function parseVideoId(url) {
  try {
    const parsed = new URL(url.trim())
    if (parsed.hostname === 'youtu.be') {
      return parsed.pathname.slice(1).split('?')[0] || null
    }
    if (['www.youtube.com', 'youtube.com', 'm.youtube.com'].includes(parsed.hostname)) {
      if (parsed.pathname === '/watch') {
        return parsed.searchParams.get('v') || null
      }
      if (parsed.pathname.startsWith('/embed/') || parsed.pathname.startsWith('/v/')) {
        return parsed.pathname.split('/')[2] || null
      }
    }
  } catch {
    // invalid URL
  }
  return null
}

export default function App() {
  const [url, setUrl] = useState('')
  const [videoId, setVideoId] = useState(null)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function handleUrlChange(e) {
    const val = e.target.value
    setUrl(val)
    setVideoId(parseVideoId(val))
    setError(null)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!question.trim() || !url.trim()) return

    const userQuestion = question.trim()
    setMessages(prev => [...prev, { role: 'user', text: userQuestion }])
    setQuestion('')
    setLoading(true)
    setError(null)

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: url, question: userQuestion }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Error ${res.status}`)
      }

      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', text: data.answer }])
    } catch (err) {
      setError(err.message)
      setMessages(prev => prev.slice(0, -1))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>YouTube Chatbot</h1>
      </header>

      <div className="url-section">
        <label htmlFor="url-input">Paste a YouTube URL:</label>
        <input
          id="url-input"
          type="text"
          value={url}
          onChange={handleUrlChange}
          placeholder="https://www.youtube.com/watch?v=..."
        />
        {videoId && (
          <img
            className="thumbnail"
            src={`https://img.youtube.com/vi/${videoId}/hqdefault.jpg`}
            alt="Video thumbnail"
          />
        )}
      </div>

      <div className="chat-section">
        {messages.length === 0 && !loading && (
          <p className="empty-state">Ask a question about the video above.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`bubble ${msg.role}`}>
            <span className="bubble-label">{msg.role === 'user' ? 'You' : 'Assistant'}</span>
            <p>{msg.text}</p>
          </div>
        ))}
        {loading && (
          <div className="bubble assistant loading">
            <span className="bubble-label">Assistant</span>
            <p>Thinking...</p>
          </div>
        )}
        {error && <p className="error">{error}</p>}
      </div>

      <form className="input-bar" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim() || !url.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

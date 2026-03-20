import { useState, useEffect, useCallback } from 'react'

export function useToast() {
  const [toasts, setToasts] = useState([])

  const toast = useCallback((message, type = 'error') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000)
  }, [])

  const dismiss = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  return { toasts, toast, dismiss }
}

export function ToastContainer({ toasts, dismiss, darkMode }) {
  const accent  = darkMode ? '#d5ff5f' : '#bce236'
  const bg      = darkMode ? '#111111' : '#ffffff'
  const text    = darkMode ? '#ffffff' : '#111111'
  const subtext = darkMode ? '#9ca3af' : '#6b7280'

  const icons = {
    error:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ff4d4d" strokeWidth="2.5" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="0.5" fill="#ff4d4d"/></svg>,
    success: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="9 12 11 14 15 10"/></svg>,
    warning: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="0.5" fill="#f59e0b"/></svg>,
    info:    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><circle cx="12" cy="8" r="0.5" fill={accent}/></svg>,
  }

  const borderColor = {
    error:   '#ff4d4d33',
    success: `${accent}44`,
    warning: '#f59e0b33',
    info:    `${accent}33`,
  }

  return (
    <div style={{
      position: 'fixed', bottom: '2rem', right: '2rem',
      zIndex: 9999, display: 'flex', flexDirection: 'column', gap: '0.75rem',
      pointerEvents: 'none',
    }}>
      {toasts.map(t => (
        <div key={t.id} style={{
          pointerEvents: 'all',
          display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
          padding: '0.9rem 1.1rem',
          backgroundColor: bg,
          border: `1px solid ${borderColor[t.type] || borderColor.error}`,
          borderRadius: '16px',
          boxShadow: darkMode
            ? '0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05)'
            : '0 8px 32px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.9)',
          backdropFilter: 'blur(12px)',
          maxWidth: '340px', minWidth: '260px',
          animation: 'toastIn 0.3s cubic-bezier(0.34,1.56,0.64,1) forwards',
          fontFamily: 'Bricolage Grotesque, sans-serif',
        }}>
          <div style={{ flexShrink: 0, marginTop: '1px' }}>{icons[t.type] || icons.error}</div>
          <div style={{ flex: 1 }}>
            <p style={{ margin: 0, fontSize: '0.85rem', color: text, lineHeight: 1.5, fontWeight: '500' }}>
              {t.message}
            </p>
          </div>
          <button onClick={() => dismiss(t.id)} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: subtext, padding: '0', flexShrink: 0, lineHeight: 1,
            display: 'flex', alignItems: 'center',
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
      ))}
      <style>{`
        @keyframes toastIn {
          0% { opacity: 0; transform: translateY(16px) scale(0.95); }
          100% { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
    </div>
  )
}
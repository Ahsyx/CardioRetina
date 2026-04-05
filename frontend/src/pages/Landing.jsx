import { useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'

function useAnimate(delay = 0) {
  const [style, setStyle] = useState({
    opacity: 0, transform: 'translateY(24px)', filter: 'blur(8px)',
    transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms`
  })
  useEffect(() => {
    const t = setTimeout(() => setStyle({
      opacity: 1, transform: 'translateY(0)', filter: 'blur(0px)',
      transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms`
    }), 50)
    return () => clearTimeout(t)
  }, [])
  return style
}

function AnatomicalHeart({ accent, size = 340 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="heartGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={accent} stopOpacity="0.25"/>
          <stop offset="100%" stopColor={accent} stopOpacity="0"/>
        </radialGradient>
        <radialGradient id="heartFill" cx="45%" cy="40%" r="60%">
          <stop offset="0%" stopColor={accent} stopOpacity="0.18"/>
          <stop offset="60%" stopColor={accent} stopOpacity="0.08"/>
          <stop offset="100%" stopColor={accent} stopOpacity="0.03"/>
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="softGlow">
          <feGaussianBlur stdDeviation="6" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <ellipse cx="150" cy="160" rx="100" ry="110" fill="url(#heartGlow)"/>
      <path d="M150 240 C150 240 60 185 48 130 C36 75 70 50 95 52 C112 53 128 62 138 75 L150 92 L162 75 C172 62 188 53 205 52 C230 50 264 75 252 130 C240 185 150 240 150 240Z" fill="url(#heartFill)" stroke={accent} strokeWidth="1.8" opacity="0.9"/>
      <path d="M205 52 C220 48 240 55 248 70 C255 85 252 105 245 118 C238 131 228 138 218 140 C208 142 198 136 192 125 C185 113 186 98 192 85 C198 72 205 52 205 52Z" fill={`${accent}12`} stroke={accent} strokeWidth="1.2" opacity="0.7"/>
      <path d="M95 52 C80 48 60 55 52 70 C45 85 48 105 55 118 C62 131 72 138 82 140 C92 142 102 136 108 125 C115 113 114 98 108 85 C102 72 95 52 95 52Z" fill={`${accent}12`} stroke={accent} strokeWidth="1.2" opacity="0.7"/>
      <path d="M150 145 C165 145 185 148 198 158 C212 168 218 182 215 198 C212 214 200 228 185 236 C170 244 155 245 150 240" fill={`${accent}10`} stroke={accent} strokeWidth="1.4" opacity="0.8"/>
      <path d="M150 145 C135 145 115 148 102 158 C88 168 82 182 85 198 C88 214 100 228 115 236 C130 244 145 245 150 240" fill={`${accent}10`} stroke={accent} strokeWidth="1.4" opacity="0.8"/>
      <path d="M150 145 L150 240" stroke={accent} strokeWidth="1" strokeDasharray="3 3" opacity="0.4"/>
      <path d="M162 75 C162 65 165 52 168 42 C171 32 175 24 178 18" stroke={accent} strokeWidth="5" strokeLinecap="round" opacity="0.8" filter="url(#glow)"/>
      <path d="M162 75 C162 65 165 52 168 42 C171 32 175 24 178 18" stroke={accent} strokeWidth="2.5" strokeLinecap="round" opacity="0.6"/>
      <path d="M178 18 C185 12 196 10 204 14 C212 18 215 26 212 34" stroke={accent} strokeWidth="4.5" strokeLinecap="round" opacity="0.7" filter="url(#glow)"/>
      <path d="M178 18 C185 12 196 10 204 14 C212 18 215 26 212 34" stroke={accent} strokeWidth="2" strokeLinecap="round" opacity="0.6"/>
      <path d="M138 75 C136 64 132 50 128 38 C124 26 120 18 116 14" stroke={accent} strokeWidth="4" strokeLinecap="round" opacity="0.65" filter="url(#glow)"/>
      <path d="M138 75 C136 64 132 50 128 38 C124 26 120 18 116 14" stroke={accent} strokeWidth="1.8" strokeLinecap="round" opacity="0.5"/>
      <path d="M116 14 C108 8 96 8 90 14" stroke={accent} strokeWidth="3" strokeLinecap="round" opacity="0.6"/>
      <path d="M116 14 C120 8 130 6 136 10" stroke={accent} strokeWidth="3" strokeLinecap="round" opacity="0.6"/>
      <path d="M218 80 C228 72 238 65 244 55 C248 48 248 40 244 34" stroke={accent} strokeWidth="4" strokeLinecap="round" opacity="0.6"/>
      <path d="M230 160 C244 162 254 168 258 178 C262 188 260 200 254 208" stroke={accent} strokeWidth="3.5" strokeLinecap="round" opacity="0.55"/>
      <path d="M155 100 C162 108 168 120 165 135 C162 150 155 160 150 165" stroke={accent} strokeWidth="1.5" strokeLinecap="round" opacity="0.5" strokeDasharray="2 2"/>
      <path d="M145 100 C138 108 132 120 135 135 C138 150 145 160 150 165" stroke={accent} strokeWidth="1.5" strokeLinecap="round" opacity="0.5" strokeDasharray="2 2"/>
      <path d="M150 165 C145 178 140 192 138 208 C136 220 138 232 142 238" stroke={accent} strokeWidth="1.2" strokeLinecap="round" opacity="0.4" strokeDasharray="2 3"/>
      <path d="M150 165 C155 178 160 192 162 208 C164 220 162 232 158 238" stroke={accent} strokeWidth="1.2" strokeLinecap="round" opacity="0.4" strokeDasharray="2 3"/>
      <ellipse cx="150" cy="118" rx="14" ry="8" stroke={accent} strokeWidth="1.2" fill={`${accent}15`} opacity="0.7"/>
      <path d="M136 118 C140 112 145 110 150 112 C155 110 160 112 164 118" stroke={accent} strokeWidth="1" opacity="0.6"/>
      <ellipse cx="133" cy="90" rx="9" ry="5" stroke={accent} strokeWidth="1" fill={`${accent}12`} opacity="0.6" transform="rotate(-20 133 90)"/>
      <ellipse cx="165" cy="90" rx="9" ry="5" stroke={accent} strokeWidth="1" fill={`${accent}12`} opacity="0.6" transform="rotate(20 165 90)"/>
      <path d="M100 90 C108 75 122 68 138 70 C145 71 150 75 150 80" stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.12"/>
      <circle cx="150" cy="165" r="4" fill={accent} opacity="0.8">
        <animate attributeName="r" values="4;7;4" dur="1.2s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.8;0.2;0.8" dur="1.2s" repeatCount="indefinite"/>
      </circle>
    </svg>
  )
}

function ECGLine({ accent }) {
  return (
    <svg viewBox="0 0 700 60" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', opacity: 0.4 }}>
      <style>{`.ecg { stroke-dasharray: 1400; stroke-dashoffset: 1400; animation: drawEcg 3s ease-in-out infinite; }
        @keyframes drawEcg { 0%{stroke-dashoffset:1400;opacity:0} 10%{opacity:1} 80%{opacity:1} 100%{stroke-dashoffset:0;opacity:0.5} }`}
      </style>
      <path className="ecg"
        d="M0 30 L80 30 L100 30 L115 30 L125 8 L135 52 L145 3 L158 57 L168 30 L185 30 L230 30 L245 30 L260 15 L270 45 L280 30 L320 30 L360 30 L395 30 L405 10 L415 50 L425 5 L438 55 L448 30 L490 30 L530 30 L575 30 L590 18 L600 42 L610 30 L700 30"
        stroke={accent} strokeWidth="2" strokeLinecap="round"/>
    </svg>
  )
}

function Landing({ darkMode, setDarkMode, purpleMode, setPurpleMode }) {
  const navigate = useNavigate()

  // ── Theme values ──────────────────────────────────────────
  const accent  = purpleMode ? '#ffe649' : darkMode ? '#d5ff5f' : '#bce236'
  const bg      = purpleMode ? '#470c98' : darkMode ? '#000000' : '#ffffff'
  const card    = purpleMode ? '#3a0a7a' : darkMode ? '#0d0d0d' : '#f8f8f8'
  const text    = purpleMode ? '#ffffff' : darkMode ? '#ffffff' : '#111111'
  const subtext = purpleMode ? '#c4a8f0' : darkMode ? '#9ca3af' : '#6b7280'
  const border  = purpleMode ? '#6020c0' : darkMode ? '#1e1e1e' : '#e5e5e5'

  const handleLogoClick = () => navigate('/')

  const a0 = useAnimate(0)
  const a1 = useAnimate(200)
  const a2 = useAnimate(400)
  const a3 = useAnimate(600)
  const a4 = useAnimate(800)
  const a5 = useAnimate(1000)
  const a6 = useAnimate(1200)

  const btnPrimary = {
    padding: '0.75rem 2rem', borderRadius: '999px',
    backgroundColor: accent, color: purpleMode ? '#470c98' : '#000',
    fontWeight: '600', border: 'none', cursor: 'pointer',
    fontSize: '0.95rem', fontFamily: 'Bricolage Grotesque, sans-serif',
    display: 'flex', alignItems: 'center', gap: '0.4rem',
    transition: 'transform 0.2s, box-shadow 0.2s',
    boxShadow: `0 0 20px ${accent}44`,
  }
  const btnSecondary = {
    padding: '0.75rem 2rem', borderRadius: '999px',
    backgroundColor: 'transparent', color: text,
    fontWeight: '500', border: `1px solid ${border}`, cursor: 'pointer',
    fontSize: '0.95rem', fontFamily: 'Bricolage Grotesque, sans-serif',
    display: 'flex', alignItems: 'center', gap: '0.4rem',
    transition: 'border-color 0.2s',
  }

  const techStack = [
    { name: 'Python', slug: 'python' }, { name: 'TensorFlow', slug: 'tensorflow' },
    { name: 'React', slug: 'react' }, { name: 'FastAPI', slug: 'fastapi' },
    { name: 'Firebase', slug: 'firebase' }, { name: 'Keras', slug: 'keras' },
    { name: 'scikit-learn', slug: 'scikitlearn' }, { name: 'NumPy', slug: 'numpy' },
  ]

  const stats = [
    { value: '88%',   label: 'Model Accuracy',     sub: 'On RFMiD test set' },
    { value: '0.91',  label: 'AUC-ROC Score',       sub: 'Area under curve' },
    { value: '1,920', label: 'Training Images',      sub: 'Fundus photographs' },
    { value: '88%',   label: 'High Risk Recall',     sub: 'Sensitivity score' },
    { value: '0.20',  label: 'Risk Threshold',       sub: 'Tuned for recall' },
    { value: '6',     label: 'Biomarkers Detected',  sub: 'HR, DR, BRVO, CRVO+' },
  ]

  const steps = [
    {
      num: '01', title: 'Upload Retinal Scan',
      desc: 'Submit a fundus photograph — JPEG or PNG. Our CV security filter validates the image before processing.',
      icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
    },
    {
      num: '02', title: 'AI Detects Biomarkers',
      desc: 'EfficientNetB1 + B0 ensemble scans for vascular changes. Grad-CAM heatmap highlights exact regions of concern on the retina.',
      icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
    },
    {
      num: '03', title: 'Get Report + Consult',
      desc: 'Receive cardiovascular risk score, findings, and recommendation. Connect with a cardiologist directly.',
      icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
    },
  ]

  return (
    <div style={{ backgroundColor: bg, color: text, minHeight: '100vh', fontFamily: 'Bricolage Grotesque, sans-serif', transition: 'background-color 0.4s, color 0.4s', overflowX: 'hidden' }}>

      <style>{`
        @keyframes marquee { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
        @keyframes floatY  { 0%,100%{transform:translateY(0) rotate(-2deg)} 50%{transform:translateY(-14px) rotate(2deg)} }
        @keyframes pulse   { 0%,100%{opacity:0.6} 50%{opacity:1} }
        @keyframes spinSlow { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
        .marquee-track { display:flex; width:max-content; animation:marquee 22s linear infinite; }
        .marquee-track:hover { animation-play-state:paused; }
        .heart-float { animation: floatY 5s ease-in-out infinite; }
        .stat-card:hover { transform: translateY(-4px) !important; border-color: var(--accent) !important; }
        .step-card:hover { transform: translateY(-6px) !important; border-color: var(--accent) !important; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 0 32px var(--accent-glow) !important; }
        .btn-secondary:hover { border-color: var(--accent) !important; }
        .logo-btn { cursor: pointer; transition: opacity 0.2s; }
        .logo-btn:hover { opacity: 0.8; }
      `}</style>

      <div style={{ display: 'none' }}>
        <style>{`:root { --accent: ${accent}; --accent-glow: ${accent}66; }`}</style>
      </div>

      {/* ── Navbar ── */}
      <nav style={{ ...a0, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.4rem 4rem', position: 'sticky', top: 0, backgroundColor: `${bg}ee`, backdropFilter: 'blur(12px)', zIndex: 100, borderBottom: `1px solid ${border}` }}>
        
        {/* Logo — clickable to toggle purple mode */}
        <div className="logo-btn" style={{ display: 'flex', alignItems: 'baseline' }} onClick={handleLogoClick}>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
        </div>
        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
        <button onClick={() => setPurpleMode(p => !p)} title="Toggle violet theme"
          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', opacity: purpleMode ? 1 : 0.5, fontSize: '1rem', transition: 'opacity 0.2s' }}
          onMouseEnter={e => e.currentTarget.style.opacity = 1}
          onMouseLeave={e => e.currentTarget.style.opacity = purpleMode ? 1 : 0.5}>✦</button>
        {/* Dark/light toggle — hidden in purple mode */}
        {!purpleMode && (
            <button onClick={() => setDarkMode(!darkMode)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', opacity: 0.6, display: 'flex' }}
              onMouseEnter={e => e.currentTarget.style.opacity = 1} onMouseLeave={e => e.currentTarget.style.opacity = 0.6}>
              {darkMode
                ? <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
                : <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>}
            </button>
          )}
          <button onClick={() => navigate('/login')} className="btn-secondary" style={btnSecondary}>Sign In</button>
          <button onClick={() => navigate('/signup')} className="btn-primary" style={btnPrimary}>Get Started →</button>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section style={{ maxWidth: '1200px', margin: '0 auto', padding: '5rem 4rem 3rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'center', minHeight: '85vh' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{ ...a1, display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.35rem 1rem', borderRadius: '999px', border: `1px solid ${accent}44`, backgroundColor: `${accent}0d`, width: 'fit-content' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: accent, display: 'inline-block', animation: 'pulse 1.5s ease-in-out infinite' }}/>
            <span style={{ fontSize: '0.78rem', color: accent, letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: '600' }}>AI Powered Medical Screening</span>
          </div>

          <h1 style={{ ...a2, fontSize: 'clamp(2.8rem, 5vw, 5rem)', fontWeight: '400', lineHeight: 1.05, letterSpacing: '-0.02em', fontFamily: 'Junicode, serif', margin: 0 }}>
            See the risk<br/>
            before <span style={{ color: accent, fontStyle: 'italic' }}>it sees you</span>
          </h1>

          <p style={{ ...a3, fontSize: '1rem', color: subtext, lineHeight: 1.9, margin: 0, maxWidth: '480px' }}>
            Early detection of cardiovascular risk, directly from the eye. CardioRetina analyzes retinal fundus images using deep learning to identify vascular biomarkers before symptoms emerge.
          </p>

          <div style={{ ...a4, display: 'flex', gap: '2rem' }}>
            {[{ v: '96.4%', l: 'AUC Score' }, { v: '91%', l: 'Recall' }, { v: '6', l: 'Biomarkers' }].map((s, i) => (
              <div key={i}>
                <div style={{ fontSize: '1.6rem', fontWeight: '700', color: accent, fontFamily: 'Junicode, serif', lineHeight: 1 }}>{s.v}</div>
                <div style={{ fontSize: '0.75rem', color: subtext, marginTop: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{s.l}</div>
              </div>
            ))}
          </div>

          <div style={{ ...a5, display: 'flex', gap: '1rem' }}>
            <button onClick={() => navigate('/signup')} className="btn-primary" style={btnPrimary}>Start Screening →</button>
            <button onClick={() => navigate('/login')} className="btn-secondary" style={btnSecondary}>Sign In</button>
          </div>
        </div>

        <div style={{ ...a6, display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
          <div style={{ position: 'absolute', width: '320px', height: '320px', borderRadius: '50%', background: `radial-gradient(circle, ${accent}22 0%, transparent 70%)`, filter: 'blur(40px)', zIndex: 0 }}/>
          <div style={{ position: 'absolute', width: '380px', height: '380px', borderRadius: '50%', border: `1px dashed ${accent}22`, zIndex: 0, animation: 'spinSlow 20s linear infinite' }}/>
          <div style={{ position: 'absolute', width: '440px', height: '440px', borderRadius: '50%', border: `1px dashed ${accent}11`, zIndex: 0, animation: 'spinSlow 30s linear infinite reverse' }}/>

          <div className="heart-float" style={{
            position: 'relative', zIndex: 1,
            backgroundColor: purpleMode ? 'rgba(255,255,255,0.06)' : darkMode ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)',
            backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
            border: `1px solid ${purpleMode ? 'rgba(255,255,255,0.15)' : darkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)'}`,
            borderRadius: '32px', padding: '2.5rem',
            boxShadow: purpleMode
              ? `0 0 0 1px ${accent}22, 0 32px 80px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1)`
              : darkMode
                ? `0 0 0 1px ${accent}11, 0 32px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06)`
                : `0 0 0 1px ${accent}22, 0 32px 80px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.8)`,
            maxWidth: '360px', width: '100%',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div>
                <p style={{ margin: 0, fontSize: '0.7rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Cardiovascular</p>
                <p style={{ margin: 0, fontSize: '1.1rem', fontWeight: '600', fontFamily: 'Junicode, serif', color: text }}>Risk Analysis</p>
              </div>
              <div style={{ padding: '0.4rem 0.8rem', borderRadius: '999px', backgroundColor: `${accent}18`, border: `1px solid ${accent}33` }}>
                <span style={{ fontSize: '0.72rem', color: accent, fontWeight: '600' }}>AI Active</span>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', margin: '0.5rem 0' }}>
              <AnatomicalHeart accent={accent} size={220}/>
            </div>
            <div style={{ margin: '0.5rem 0', padding: '0.8rem', borderRadius: '12px', backgroundColor: purpleMode ? 'rgba(255,255,255,0.05)' : darkMode ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)', border: `1px solid ${border}` }}>
              <p style={{ margin: '0 0 0.3rem', fontSize: '0.65rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Live ECG Simulation</p>
              <ECGLine accent={accent}/>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '1rem' }}>
              {[
                { label: 'Model', value: 'EfficientNetB1 + B0' },
                { label: 'AUC Score', value: '0.9642' },
                { label: 'Threshold', value: '0.20 (High Recall)' },
              ].map((row, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: `1px solid ${border}` }}>
                  <span style={{ fontSize: '0.78rem', color: subtext }}>{row.label}</span>
                  <span style={{ fontSize: '0.78rem', fontWeight: '600', color: accent }}>{row.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── ECG divider ── */}
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 4rem' }}>
        <ECGLine accent={accent}/>
      </div>

      {/* ── Stats Grid ── */}
      <section style={{ maxWidth: '1200px', margin: '0 auto', padding: '5rem 4rem 2rem' }}>
        <div style={{ marginBottom: '3rem' }}>
          <p style={{ margin: '0 0 0.5rem', fontSize: '0.75rem', color: accent, textTransform: 'uppercase', letterSpacing: '0.12em', fontWeight: '600' }}>Model Performance</p>
          <h2 style={{ margin: 0, fontSize: '2.2rem', fontFamily: 'Junicode, serif', fontWeight: '400', color: text }}>
            Built on <span style={{ color: accent, fontStyle: 'italic' }}>Clinical Evidence</span>
          </h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1px', backgroundColor: border, borderRadius: '24px', overflow: 'hidden', border: `1px solid ${border}` }}>
          {stats.map((s, i) => (
            <div key={i} className="stat-card" style={{ backgroundColor: card, padding: '2rem', transition: 'transform 0.2s, border-color 0.2s', cursor: 'default', borderBottom: i < 3 ? `1px solid ${border}` : 'none' }}>
              <div style={{ fontSize: '2.4rem', fontWeight: '700', color: accent, fontFamily: 'Junicode, serif', lineHeight: 1, marginBottom: '0.4rem' }}>{s.value}</div>
              <div style={{ fontSize: '0.9rem', fontWeight: '600', marginBottom: '0.25rem', color: text }}>{s.label}</div>
              <div style={{ fontSize: '0.78rem', color: subtext }}>{s.sub}</div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '1rem', padding: '1.2rem 2rem', borderRadius: '16px', border: `1px solid ${border}`, backgroundColor: card, display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          {[
            { l: 'Architecture', v: 'EfficientNetB1 + B0 Ensemble' },
            { l: 'Dataset', v: 'RFMiD — Retinal Fundus Multi-Disease' },
            { l: 'Framework', v: 'TensorFlow / Keras 3' },
            { l: 'Input Size', v: '240 × 240 px' },
          ].map((item, i) => (
            <div key={i}>
              <div style={{ fontSize: '0.68rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.2rem' }}>{item.l}</div>
              <div style={{ fontSize: '0.85rem', fontWeight: '600', color: text }}>{item.v}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── How It Works ── */}
      <section style={{ maxWidth: '1200px', margin: '0 auto', padding: '4rem 4rem' }}>
        <div style={{ marginBottom: '3rem' }}>
          <p style={{ margin: '0 0 0.5rem', fontSize: '0.75rem', color: accent, textTransform: 'uppercase', letterSpacing: '0.12em', fontWeight: '600' }}>Process</p>
          <h2 style={{ margin: 0, fontSize: '2.2rem', fontFamily: 'Junicode, serif', fontWeight: '400', color: text }}>
            How <span style={{ color: accent, fontStyle: 'italic' }}>CardioRetina</span> Works
          </h2>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
          {steps.map((step, i) => (
            <div key={i} className="step-card" style={{
              backgroundColor: purpleMode ? 'rgba(255,255,255,0.05)' : darkMode ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
              backdropFilter: 'blur(10px)', border: `1px solid ${border}`,
              borderRadius: '24px', padding: '2rem',
              transition: 'transform 0.25s, border-color 0.25s', cursor: 'default',
              boxShadow: purpleMode ? 'inset 0 1px 0 rgba(255,255,255,0.08)' : darkMode ? 'inset 0 1px 0 rgba(255,255,255,0.04)' : 'inset 0 1px 0 rgba(255,255,255,0.9)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                <div style={{ width: '46px', height: '46px', borderRadius: '14px', backgroundColor: `${accent}15`, border: `1px solid ${accent}33`, color: accent, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {step.icon}
                </div>
                <span style={{ fontSize: '2.5rem', fontWeight: '700', color: border, fontFamily: 'Junicode, serif', lineHeight: 1 }}>{step.num}</span>
              </div>
              <h3 style={{ margin: '0 0 0.6rem', fontSize: '1rem', fontWeight: '600', color: text }}>{step.title}</h3>
              <p style={{ margin: 0, fontSize: '0.85rem', color: subtext, lineHeight: 1.75 }}>{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem 4rem 5rem' }}>
        <div style={{
          borderRadius: '28px', padding: '3.5rem',
          background: purpleMode
            ? `linear-gradient(135deg, ${accent}15 0%, transparent 60%)`
            : darkMode
              ? `linear-gradient(135deg, ${accent}0d 0%, transparent 60%)`
              : `linear-gradient(135deg, ${accent}18 0%, transparent 60%)`,
          border: `1px solid ${accent}33`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '2rem',
          backdropFilter: 'blur(10px)', boxShadow: `0 0 60px ${accent}0d`,
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.8rem' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill={accent} stroke="none"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
              <span style={{ fontSize: '0.75rem', color: accent, textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: '600' }}>Early Detection Saves Lives</span>
            </div>
            <h2 style={{ margin: '0 0 0.6rem', fontSize: '1.8rem', fontFamily: 'Junicode, serif', fontWeight: '400', color: text }}>
              Your eyes reveal your <span style={{ color: accent, fontStyle: 'italic' }}>heart's health</span>
            </h2>
            <p style={{ margin: 0, color: subtext, fontSize: '0.9rem', maxWidth: '480px', lineHeight: 1.7 }}>
              Cardiovascular disease is the leading cause of death globally. Retinal imaging offers a non-invasive window into vascular health — enabling early intervention before symptoms appear.
            </p>
          </div>
          <button onClick={() => navigate('/signup')} className="btn-primary" style={{ ...btnPrimary, padding: '1rem 2.5rem', fontSize: '1rem', flexShrink: 0 }}>
            Start Free Screening →
          </button>
        </div>
      </section>

      {/* ── Marquee ── */}
      <section style={{ padding: '2rem 0 5rem', overflow: 'hidden' }}>
        <p style={{ textAlign: 'center', fontSize: '0.75rem', color: subtext, letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: '500', marginBottom: '2rem' }}>Built with</p>
        <div style={{ position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to right, ${bg}, transparent)`, zIndex: 2 }}/>
          <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to left, ${bg}, transparent)`, zIndex: 2 }}/>
          <div className="marquee-track">
            {[...techStack, ...techStack].map((tech, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', margin: '0 2.5rem', opacity: 0.4, transition: 'opacity 0.2s', whiteSpace: 'nowrap' }}
                onMouseEnter={e => e.currentTarget.style.opacity = 1}
                onMouseLeave={e => e.currentTarget.style.opacity = 0.4}>
                <img src={`https://cdn.simpleicons.org/${tech.slug}`} alt={tech.name} width="20" height="20" style={{ filter: purpleMode || darkMode ? 'brightness(0) invert(1)' : 'brightness(0)' }}/>
                <span style={{ fontSize: '0.92rem', color: text, fontWeight: '500' }}>{tech.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer style={{ textAlign: 'center', padding: '2rem', borderTop: `1px solid ${border}`, color: subtext, fontSize: '0.82rem' }}>
        © 2026 CardioRetina — AI Powered Cardiovascular Screening
      </footer>

    </div>
  )
}

export default Landing
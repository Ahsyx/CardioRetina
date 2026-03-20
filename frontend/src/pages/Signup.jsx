import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { auth, db } from '../firebase'
import { createUserWithEmailAndPassword } from 'firebase/auth'
import { useToast, ToastContainer } from '../components/Toast'
import { doc, setDoc } from 'firebase/firestore'

function useAnimate(delay = 0) {
  const [style, setStyle] = useState({
    opacity: 0, transform: 'translateY(24px)', filter: 'blur(8px)',
    transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms`
  })
  useEffect(() => {
    const t = setTimeout(() => {
      setStyle({ opacity: 1, transform: 'translateY(0)', filter: 'blur(0px)', transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms` })
    }, 50)
    return () => clearTimeout(t)
  }, [])
  return style
}

function Signup({ darkMode, setDarkMode }) {
  const navigate = useNavigate()
  const [role, setRole] = useState(null)
  const [formData, setFormData] = useState({ name: '', email: '', password: '', confirmPassword: '', license: null })
  const [loading, setLoading] = useState(false)
  const { toasts, toast, dismiss } = useToast()

  const accent  = darkMode ? '#d5ff5f' : '#bce236'
  const bg      = darkMode ? '#000000' : '#ffffff'
  const card    = darkMode ? '#1a1a1a' : '#f0f0ed'
  const text    = darkMode ? '#ffffff' : '#111111'
  const subtext = darkMode ? '#9ca3af' : '#6b7280'
  const border  = darkMode ? '#222222' : '#e5e5e5'

  const navAnim         = useAnimate(0)
  const badgeAnim       = useAnimate(200)
  const titleAnim       = useAnimate(350)
  const descAnim        = useAnimate(500)
  const formWrapperAnim = useAnimate(650)
  const techAnim        = useAnimate(800)

  const techStack = [
    { name: 'Python',      slug: 'python'      },
    { name: 'TensorFlow',  slug: 'tensorflow'  },
    { name: 'React',       slug: 'react'       },
    { name: 'FastAPI',     slug: 'fastapi'     },
    { name: 'Firebase',    slug: 'firebase'    },
    { name: 'Keras',       slug: 'keras'       },
    { name: 'scikit-learn',slug: 'scikitlearn' },
    { name: 'NumPy',       slug: 'numpy'       },
  ]

  const btnPrimary = {
    padding: '0.7rem 1.8rem', borderRadius: '999px',
    backgroundColor: darkMode ? '#ffffff' : '#111111',
    color: darkMode ? '#000000' : '#ffffff',
    fontWeight: '500', border: 'none', cursor: 'pointer',
    fontSize: '0.95rem', fontFamily: 'Bricolage Grotesque, sans-serif',
    display: 'flex', alignItems: 'center', gap: '0.4rem'
  }

  const btnSecondary = {
    padding: '0.7rem 1.8rem', borderRadius: '999px',
    backgroundColor: darkMode ? '#1a1a1a' : '#f0f0ed',
    color: text, fontWeight: '500', border: 'none', cursor: 'pointer',
    fontSize: '0.95rem', fontFamily: 'Bricolage Grotesque, sans-serif',
    display: 'flex', alignItems: 'center', gap: '0.4rem'
  }

  const inputStyle = {
    width: '100%', padding: '0.8rem 1rem', borderRadius: '12px',
    backgroundColor: bg, border: `1px solid ${border}`, color: text,
    fontFamily: 'Bricolage Grotesque, sans-serif', marginBottom: '1rem',
    outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.2s'
  }

  const handleSignup = async (e) => {
    e.preventDefault()
    if (!role) return toast('Please select a role.', 'warning')
    if (formData.password !== formData.confirmPassword) return toast('Passwords do not match.', 'error')

    setLoading(true)
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password)
      const user = userCredential.user

      await setDoc(doc(db, 'users', user.uid), {
        uid: user.uid,
        name: formData.name,
        email: formData.email,
        role: role,
        verificationStatus: role === 'doctor' ? 'Pending (Local Verification)' : 'Verified',
        createdAt: new Date().toISOString()
      })

      navigate(role === 'doctor' ? '/dashboard/doctor' : '/dashboard/patient')
    } catch (err) {
      toast(err.message.replace('Firebase: ', '').replace(/\s*\(auth\/.*?\)/, ''), 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      backgroundColor: bg, color: text, minHeight: '100vh',
      fontFamily: 'Bricolage Grotesque, sans-serif',
      transition: 'background-color 0.3s ease, color 0.3s ease',
      overflowX: 'hidden'
    }}>

      <style>{`
        @keyframes dynamicReveal {
          0% { opacity: 0; transform: translateY(24px); filter: blur(8px); }
          100% { opacity: 1; transform: translateY(0); filter: blur(0px); }
        }
        .stagger-1 { opacity: 0; animation: dynamicReveal 0.8s ease 0.1s forwards; }
        .stagger-2 { opacity: 0; animation: dynamicReveal 0.8s ease 0.2s forwards; }
        .stagger-3 { opacity: 0; animation: dynamicReveal 0.8s ease 0.3s forwards; }
        .stagger-4 { opacity: 0; animation: dynamicReveal 0.8s ease 0.4s forwards; }
        .stagger-5 { opacity: 0; animation: dynamicReveal 0.8s ease 0.5s forwards; }
        .stagger-6 { opacity: 0; animation: dynamicReveal 0.8s ease 0.6s forwards; }
        .stagger-7 { opacity: 0; animation: dynamicReveal 0.8s ease 0.7s forwards; }
        @keyframes marquee { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .marquee-track { display: flex; width: max-content; animation: marquee 20s linear infinite; }
        .marquee-track:hover { animation-play-state: paused; }
        .role-card:hover { border-color: ${accent} !important; transform: translateY(-2px); }
      `}</style>

      {/* Navbar */}
      <nav style={{
        ...navAnim, display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '1.5rem 4rem', position: 'sticky', top: 0,
        backgroundColor: bg, zIndex: 100, width: '100%', boxSizing: 'border-box'
      }}>
        <div onClick={() => navigate('/')} style={{ display: 'flex', alignItems: 'baseline', cursor: 'pointer' }}>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
        </div>
        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
          <button onClick={() => setDarkMode(!darkMode)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', display: 'flex', alignItems: 'center', opacity: 0.7, transition: 'opacity 0.2s' }} onMouseEnter={e => e.currentTarget.style.opacity = 1} onMouseLeave={e => e.currentTarget.style.opacity = 0.7}>
            {darkMode ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
            )}
          </button>
          <button onClick={() => navigate('/')} style={btnSecondary}>Home</button>
          <button onClick={() => navigate('/login')} style={btnPrimary}>Sign In →</button>
        </div>
      </nav>

      {/* Hero / Form Section */}
      <section style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', padding: '6rem 2rem 3rem', gap: '2rem' }}>

        <div style={{ ...badgeAnim, display: 'inline-block', padding: '0.35rem 1rem', borderRadius: '999px', border: `1px solid ${border}`, fontSize: '0.8rem', color: subtext, letterSpacing: '0.05em', textTransform: 'uppercase', fontWeight: '500' }}>
          AI Powered Medical Screening
        </div>

        <h2 style={{ ...titleAnim, margin: '-1rem 0 0 0', display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: 'clamp(2.5rem, 5vw, 4rem)', fontWeight: '500', fontFamily: 'Bricolage Grotesque, sans-serif' }}>Join</span>
          <div style={{ display: 'flex', alignItems: 'baseline' }}>
            <span style={{ fontSize: 'clamp(3rem, 6vw, 4.5rem)', fontWeight: '400', fontStyle: 'normal', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
            <span style={{ fontSize: 'clamp(3rem, 6vw, 4.5rem)', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
          </div>
        </h2>

        <p style={{ ...descAnim, fontSize: '1.1rem', color: subtext, maxWidth: '600px', lineHeight: 1.9, fontWeight: '400', marginTop: '-1rem' }}>
          Empowering heart health through retinal intelligence.
        </p>

        <div style={{ ...formWrapperAnim, width: '100%', maxWidth: '480px' }}>

          {/* Role selection */}
          {!role ? (
            <div style={{ display: 'flex', gap: '1rem' }}>
              <div className="stagger-1 role-card" onClick={() => setRole('doctor')}
                style={{ flex: 1, padding: '2.5rem 1rem', borderRadius: '24px', backgroundColor: card, border: `1px solid ${border}`, textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s', color: text }}>
                <div style={{ marginBottom: '1.5rem' }}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="1.5">
                    <rect x="8" y="2" width="8" height="4" rx="1"/><rect x="4" y="4" width="16" height="18" rx="2"/>
                    <path d="M9 14h6" stroke={accent}/><path d="M12 11v6" stroke={accent}/>
                  </svg>
                </div>
                <div style={{ fontWeight: '600' }}>Clinical User</div>
              </div>

              <div className="stagger-2 role-card" onClick={() => setRole('patient')}
                style={{ flex: 1, padding: '2.5rem 1rem', borderRadius: '24px', backgroundColor: card, border: `1px solid ${border}`, textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s', color: text }}>
                <div style={{ marginBottom: '1.5rem' }}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="1.5">
                    <circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 0 0-16 0" stroke={accent}/>
                  </svg>
                </div>
                <div style={{ fontWeight: '600' }}>Patient</div>
              </div>
            </div>

          ) : (
            <form onSubmit={handleSignup} style={{ backgroundColor: card, padding: '2.5rem 2rem', borderRadius: '24px', border: `1px solid ${border}`, textAlign: 'left' }}>

              <button type="button" className="stagger-1" onClick={() => setRole(null)}
                style={{ background: 'none', border: 'none', color: accent, cursor: 'pointer', marginBottom: '1.5rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '0.4rem', fontFamily: 'inherit' }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
                Back to roles
              </button>

              <div className="stagger-2">
                <input type="text" placeholder="Full Name" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, name: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-3">
                <input type="email" placeholder="Email Address" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, email: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-4">
                <input type="password" placeholder="Password" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, password: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-5">
                <input type="password" placeholder="Confirm Password" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, confirmPassword: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              {role === 'doctor' && (
                <div className="stagger-6" style={{ marginBottom: '1rem' }}>
                  <p style={{ fontSize: '0.85rem', color: subtext, marginBottom: '0.5rem' }}>Medical License Upload</p>
                  <input type="file" accept="image/*,.pdf" style={inputStyle}
                    onChange={e => setFormData({ ...formData, license: e.target.files[0] })}
                    onFocus={e => e.currentTarget.style.borderColor = accent}
                    onBlur={e => e.currentTarget.style.borderColor = border} />
                  <p style={{ fontSize: '0.75rem', color: subtext, marginTop: '-0.5rem', marginBottom: '0.5rem' }}>
                    Verification is handled locally. Upload for record-keeping only.
                  </p>
                </div>
              )}

              <div className={role === 'doctor' ? 'stagger-7' : 'stagger-6'}>
                <button type="submit" disabled={loading}
                  style={{ width: '100%', padding: '1rem', borderRadius: '999px', backgroundColor: darkMode ? '#ffffff' : '#111111', color: darkMode ? '#000000' : '#ffffff', border: 'none', fontWeight: '600', cursor: loading ? 'not-allowed' : 'pointer', fontFamily: 'inherit', fontSize: '1rem', opacity: loading ? 0.7 : 1, transition: 'opacity 0.2s' }}>
                  {loading ? 'Creating account...' : `Sign Up as ${role === 'doctor' ? 'Clinical User' : 'Patient'}`}
                </button>
              </div>

              <p style={{ textAlign: 'center', marginTop: '1.2rem', fontSize: '0.85rem', color: subtext }}>
                Already have an account?{' '}
                <span onClick={() => navigate('/login')} style={{ color: text, cursor: 'pointer', fontWeight: '500', textDecoration: 'underline', textDecorationColor: accent, textUnderlineOffset: '4px' }}>Sign in</span>
              </p>
            </form>
          )}
        </div>
      </section>

      {/* Built With — Marquee */}
      <section style={{ ...techAnim, padding: '2rem 0 4rem', textAlign: 'center', overflow: 'hidden', width: '100%' }}>
        <p style={{ fontSize: '0.8rem', color: subtext, letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: '500', marginBottom: '2.5rem' }}>Built with</p>
        <div style={{ overflow: 'hidden', position: 'relative' }}>
          <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to right, ${bg}, transparent)`, zIndex: 2, pointerEvents: 'none' }}/>
          <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to left, ${bg}, transparent)`, zIndex: 2, pointerEvents: 'none' }}/>
          <div className="marquee-track">
            {[...techStack, ...techStack].map((tech, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', margin: '0 2.5rem', opacity: 0.4, transition: 'opacity 0.2s', cursor: 'default', whiteSpace: 'nowrap' }}
                onMouseEnter={e => e.currentTarget.style.opacity = 1}
                onMouseLeave={e => e.currentTarget.style.opacity = 0.4}>
                <img src={`https://cdn.simpleicons.org/${tech.slug}`} alt={tech.name} width="22" height="22" style={{ filter: darkMode ? 'brightness(0) invert(1)' : 'brightness(0)' }} />
                <span style={{ fontSize: '0.95rem', color: text, fontWeight: '500', fontFamily: 'Bricolage Grotesque, sans-serif' }}>{tech.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ textAlign: 'center', padding: '2.5rem', borderTop: `1px solid ${border}`, color: subtext, fontSize: '0.85rem', width: '100%', boxSizing: 'border-box' }}>
        © 2026 CardioRetina — AI Powered Cardiovascular Screening
      </footer>
      <ToastContainer toasts={toasts} dismiss={dismiss} darkMode={darkMode} />
    </div>
  )
}

export default Signup
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { auth, db } from '../firebase'
import { signInWithEmailAndPassword } from 'firebase/auth'
import { useToast, ToastContainer } from './Toast'
import { doc, getDoc } from 'firebase/firestore'

function useAnimate(delay = 0) {
  const [style, setStyle] = useState({
    opacity: 0,
    transform: 'translateY(24px)',
    filter: 'blur(8px)',
    transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms`
  })
  useEffect(() => {
    const t = setTimeout(() => {
      setStyle({
        opacity: 1,
        transform: 'translateY(0)',
        filter: 'blur(0px)',
        transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms`
      })
    }, 50)
    return () => clearTimeout(t)
  }, [])
  return style
}

function Login({ darkMode, setDarkMode }) {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({ email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const { toasts, toast, dismiss } = useToast()

  const accent = darkMode ? '#d5ff5f' : '#bce236'
  const bg = darkMode ? '#000000' : '#ffffff'
  const card = darkMode ? '#1a1a1a' : '#f0f0ed'
  const text = darkMode ? '#ffffff' : '#111111'
  const subtext = darkMode ? '#9ca3af' : '#6b7280'
  const border = darkMode ? '#222222' : '#e5e5e5'

  const navAnim = useAnimate(0)
  const badgeAnim = useAnimate(200)
  const titleAnim = useAnimate(350)
  const descAnim = useAnimate(500)
  const formWrapperAnim = useAnimate(650)
  const techAnim = useAnimate(800)

  const techStack = [
    { name: 'Python', slug: 'python' }, { name: 'TensorFlow', slug: 'tensorflow' },
    { name: 'React', slug: 'react' }, { name: 'FastAPI', slug: 'fastapi' },
    { name: 'Firebase', slug: 'firebase' }, { name: 'Keras', slug: 'keras' },
    { name: 'scikit-learn', slug: 'scikitlearn' }, { name: 'NumPy', slug: 'numpy' },
  ]

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

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const userCredential = await signInWithEmailAndPassword(auth, formData.email, formData.password)
      const user = userCredential.user
      const docRef = doc(db, "users", user.uid)
      const docSnap = await getDoc(docRef)
      if (docSnap.exists()) {
        const userData = docSnap.data()
        if (userData.role === 'doctor') {
          navigate('/dashboard/doctor')
        } else {
          navigate('/dashboard/patient')
        }
      } else {
        toast('User record not found. Please sign up.', 'error')
      }
    } catch (err) {
      toast('Invalid email or password.', 'error')
      console.error(err)
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
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .marquee-track {
          display: flex; width: max-content;
          animation: marquee 20s linear infinite;
        }
        .marquee-track:hover { animation-play-state: paused; }
      `}</style>

      {/* Navbar */}
      <nav style={{
        ...navAnim, display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '1.5rem 4rem', position: 'sticky', top: 0,
        backgroundColor: bg, zIndex: 100, width: '100%', boxSizing: 'border-box'
      }}>
        <div onClick={() => navigate('/')} style={{ display: 'flex', alignItems: 'baseline', cursor: 'pointer' }}>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontStyle: 'normal', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
        </div>
        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
          <button onClick={() => setDarkMode(!darkMode)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', opacity: 0.7, transition: 'opacity 0.2s' }} onMouseEnter={e => e.currentTarget.style.opacity = 1} onMouseLeave={e => e.currentTarget.style.opacity = 0.7}>
            {darkMode ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
            )}
          </button>
          <button onClick={() => navigate('/')} style={btnSecondary}>Home</button>
        </div>
      </nav>

      {/* Main Form Section */}
      <section style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        textAlign: 'center', padding: '6rem 2rem 3rem', gap: '2rem', maxWidth: '100%'
      }}>
        <div style={{ ...badgeAnim, display: 'inline-block', padding: '0.35rem 1rem', borderRadius: '999px', border: `1px solid ${border}`, fontSize: '0.8rem', color: subtext, letterSpacing: '0.05em', textTransform: 'uppercase', fontWeight: '500' }}>
          Secure Access
        </div>

        <h2 style={{ ...titleAnim, margin: '-1rem 0 0 0', display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: 'clamp(2.5rem, 5vw, 4rem)', fontWeight: '500', fontFamily: 'Bricolage Grotesque, sans-serif' }}>Welcome</span>
          <span style={{ fontSize: 'clamp(2.5rem, 5vw, 4rem)', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Back</span>
        </h2>

        <p style={{ ...descAnim, fontSize: '1.1rem', color: subtext, maxWidth: '600px', lineHeight: 1.9, fontWeight: '400', marginTop: '-1rem' }}>
          Sign in to view your recent retinal scans and cardiovascular risk reports.
        </p>

        <div style={{ ...formWrapperAnim, width: '100%', maxWidth: '400px' }}>
          <form onSubmit={handleLogin} style={{ backgroundColor: card, padding: '2.5rem 2rem', borderRadius: '24px', border: `1px solid ${border}`, textAlign: 'left' }}>
            <div className="stagger-1">
              <input type="email" placeholder="Email Address" style={inputStyle} required onChange={e => setFormData({...formData, email: e.target.value})} onFocus={e => e.currentTarget.style.borderColor = accent} onBlur={e => e.currentTarget.style.borderColor = border}/>
            </div>
            <div className="stagger-2">
              <input type="password" placeholder="Password" style={inputStyle} required onChange={e => setFormData({...formData, password: e.target.value})} onFocus={e => e.currentTarget.style.borderColor = accent} onBlur={e => e.currentTarget.style.borderColor = border}/>
            </div>
            <div className="stagger-3">
              <button type="submit" disabled={loading} style={{ width: '100%', padding: '1rem', borderRadius: '999px', backgroundColor: darkMode ? '#ffffff' : '#111111', color: darkMode ? '#000000' : '#ffffff', border: 'none', fontWeight: '600', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, fontSize: '1rem', marginTop: '0.5rem', fontFamily: 'Bricolage Grotesque, sans-serif' }}>
                {loading ? 'Authenticating...' : 'Sign In'}
              </button>
            </div>
            <div className="stagger-4" style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.9rem', color: subtext }}>
              Don't have an account?{' '}
              <span onClick={() => navigate('/signup')} style={{ color: text, cursor: 'pointer', fontWeight: '500', textDecoration: 'underline', textDecorationColor: accent, textUnderlineOffset: '4px' }}>
                Sign Up
              </span>
            </div>
          </form>
        </div>
      </section>

      {/* Built With - Marquee */}
      <section style={{ ...techAnim, padding: '2rem 0 4rem', textAlign: 'center', overflow: 'hidden', width: '100%' }}>
        <p style={{ fontSize: '0.8rem', color: subtext, letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: '500', marginBottom: '2.5rem' }}>Built with</p>
        <div style={{ overflow: 'hidden', position: 'relative' }}>
          <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to right, ${bg}, transparent)`, zIndex: 2, pointerEvents: 'none' }}/>
          <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to left, ${bg}, transparent)`, zIndex: 2, pointerEvents: 'none' }}/>
          <div className="marquee-track">
            {[...techStack, ...techStack].map((tech, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', margin: '0 2.5rem', opacity: 0.4, transition: 'opacity 0.2s', cursor: 'default', whiteSpace: 'nowrap' }} onMouseEnter={e => e.currentTarget.style.opacity = 1} onMouseLeave={e => e.currentTarget.style.opacity = 0.4}>
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

export default Login
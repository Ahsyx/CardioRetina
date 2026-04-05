import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { auth, db } from '../firebase'
import { createUserWithEmailAndPassword, signInWithPopup, GoogleAuthProvider } from 'firebase/auth'
import { useToast, ToastContainer } from '../components/Toast'
import { doc, setDoc, getDoc } from 'firebase/firestore'
  import { useAuth } from '../context/AuthContext'

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

const PatientIcon = ({ size = 26, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 0 0-16 0"/>
  </svg>
)

const DoctorIcon = ({ size = 26, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
    <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/>
    <circle cx="20" cy="10" r="2"/>
  </svg>
)

const EmailIcon = ({ size = 16, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="4" width="20" height="16" rx="2"/>
    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
  </svg>
)

function GooglePermissionDialog({ onConfirm, onCancel, accent, bg, text, subtext, border, purpleMode, darkMode }) {
  const permissions = [
    { icon: <PatientIcon size={16} color={subtext}/>, label: 'Your name',          desc: 'To personalise your account' },
    { icon: <EmailIcon   size={16} color={subtext}/>, label: 'Your email address', desc: 'To identify and contact you'  },
  ]
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 1000, backgroundColor: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}>
      <div style={{ backgroundColor: purpleMode ? '#3a0a7a' : darkMode ? '#111111' : '#ffffff', border: `1px solid ${border}`, borderRadius: '28px', padding: '2.5rem 2rem', maxWidth: '400px', width: '100%', boxShadow: purpleMode ? `0 0 0 1px ${accent}22, 0 40px 100px rgba(0,0,0,0.7)` : darkMode ? '0 40px 100px rgba(0,0,0,0.8)' : '0 40px 100px rgba(0,0,0,0.15)', textAlign: 'center', animation: 'dialogIn 0.3s cubic-bezier(0.34,1.56,0.64,1) forwards' }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline', gap: '0.1rem', marginBottom: '1.5rem' }}>
          <span style={{ fontSize: '1.2rem', fontWeight: '400', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
          <span style={{ fontSize: '1.2rem', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.2rem' }}>
          <div style={{ width: '60px', height: '60px', borderRadius: '50%', backgroundColor: purpleMode ? 'rgba(255,255,255,0.08)' : darkMode ? '#1a1a1a' : '#f5f5f5', border: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="30" height="30" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
          </div>
        </div>
        <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.1rem', fontWeight: '600', color: text, fontFamily: 'Bricolage Grotesque, sans-serif' }}>Sign up with Google</h3>
        <p style={{ margin: '0 0 1.5rem', fontSize: '0.85rem', color: subtext, lineHeight: 1.7 }}>
          CardioRetina will request access to your <strong style={{ color: text }}>name</strong> and <strong style={{ color: text }}>email address</strong> to create your account.
        </p>
        <div style={{ backgroundColor: purpleMode ? 'rgba(255,255,255,0.05)' : darkMode ? '#0d0d0d' : '#f8f8f8', borderRadius: '14px', padding: '0.8rem 1rem', marginBottom: '1.5rem', textAlign: 'left' }}>
          {permissions.map((p, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.5rem 0', borderBottom: i === 0 ? `1px solid ${border}` : 'none' }}>
              {p.icon}
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '0.82rem', fontWeight: '600', color: text }}>{p.label}</div>
                <div style={{ fontSize: '0.72rem', color: subtext }}>{p.desc}</div>
              </div>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button onClick={onCancel} style={{ flex: 1, padding: '0.8rem', borderRadius: '999px', backgroundColor: 'transparent', border: `1px solid ${border}`, color: text, cursor: 'pointer', fontFamily: 'Bricolage Grotesque, sans-serif', fontWeight: '500', fontSize: '0.9rem' }} onMouseEnter={e => e.currentTarget.style.opacity = 0.7} onMouseLeave={e => e.currentTarget.style.opacity = 1}>Cancel</button>
          <button onClick={onConfirm} style={{ flex: 1, padding: '0.8rem', borderRadius: '999px', backgroundColor: accent, border: 'none', color: purpleMode ? '#470c98' : '#000', cursor: 'pointer', fontFamily: 'Bricolage Grotesque, sans-serif', fontWeight: '600', fontSize: '0.9rem', boxShadow: `0 0 20px ${accent}44` }} onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'} onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>Continue</button>
        </div>
        <p style={{ marginTop: '1rem', fontSize: '0.7rem', color: subtext }}>A Google sign-in popup will open after you continue.</p>
      </div>
    </div>
  )
}

function Signup({ darkMode, setDarkMode, purpleMode, setPurpleMode }) {
  const navigate = useNavigate()
  const location = useLocation()

  const [step, setStep]         = useState('role')
  const [role, setRole]         = useState(null)
  const [formData, setFormData] = useState({ name: '', email: '', password: '', confirmPassword: '', license: null, dob: '' })
  const [loading, setLoading]   = useState(false)
  const [googleLoading, setGoogleLoading]   = useState(false)
  const [showGoogleDialog, setShowGoogleDialog] = useState(false)
  const [googleUser, setGoogleUser] = useState(null)
  const { toasts, toast, dismiss } = useToast()
  const { refreshUserData } = useAuth()

  // Max DOB = 18 years ago from today
  const maxDob = (() => {
    const d = new Date()
    d.setFullYear(d.getFullYear() - 18)
    return d.toISOString().split('T')[0]
  })()

  useEffect(() => {
    if (location.state?.googleUser) {
      const gu = location.state.googleUser
      setGoogleUser(gu)
      setFormData(prev => ({ ...prev, name: gu.displayName || '', email: gu.email || '' }))
      setStep('google-profile')
    }
  }, [])

  const accent  = purpleMode ? '#ffe649' : darkMode ? '#d5ff5f' : '#bce236'
  const bg      = purpleMode ? '#470c98' : darkMode ? '#000000' : '#ffffff'
  const card    = purpleMode ? '#3a0a7a' : darkMode ? '#1a1a1a' : '#f0f0ed'
  const text    = purpleMode ? '#ffffff' : darkMode ? '#ffffff' : '#111111'
  const subtext = purpleMode ? '#c4a8f0' : darkMode ? '#9ca3af' : '#6b7280'
  const border  = purpleMode ? '#6020c0' : darkMode ? '#222222' : '#e5e5e5'

  const navAnim         = useAnimate(0)
  const badgeAnim       = useAnimate(200)
  const titleAnim       = useAnimate(350)
  const descAnim        = useAnimate(500)
  const formWrapperAnim = useAnimate(650)
  const techAnim        = useAnimate(800)

  const techStack = [
    { name: 'Python', slug: 'python' }, { name: 'TensorFlow', slug: 'tensorflow' },
    { name: 'React', slug: 'react' }, { name: 'FastAPI', slug: 'fastapi' },
    { name: 'Firebase', slug: 'firebase' }, { name: 'Keras', slug: 'keras' },
    { name: 'scikit-learn', slug: 'scikitlearn' }, { name: 'NumPy', slug: 'numpy' },
  ]

  const inputStyle = {
    width: '100%', padding: '0.8rem 1rem', borderRadius: '12px',
    backgroundColor: purpleMode ? 'rgba(255,255,255,0.08)' : bg,
    border: `1px solid ${border}`, color: text,
    fontFamily: 'Bricolage Grotesque, sans-serif', marginBottom: '1rem',
    outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.2s', fontSize: '0.95rem',
  }
  const btnPrimary = {
    padding: '0.75rem 2rem', borderRadius: '999px',
    backgroundColor: accent, color: purpleMode ? '#470c98' : '#000',
    fontWeight: '600', border: 'none', cursor: 'pointer',
    fontSize: '0.95rem', fontFamily: 'Bricolage Grotesque, sans-serif',
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
    transition: 'transform 0.2s, box-shadow 0.2s', boxShadow: `0 0 20px ${accent}44`,
  }
  const btnSecondary = {
    padding: '0.7rem 1.8rem', borderRadius: '999px',
    backgroundColor: purpleMode ? 'rgba(255,255,255,0.1)' : darkMode ? '#1a1a1a' : '#f0f0ed',
    color: text, fontWeight: '500',
    border: purpleMode ? `1px solid ${border}` : 'none',
    cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'Bricolage Grotesque, sans-serif',
    display: 'flex', alignItems: 'center', gap: '0.4rem',
  }
  const cardStyle = {
    backgroundColor: purpleMode ? 'rgba(255,255,255,0.06)' : card,
    padding: '2.5rem 2rem', borderRadius: '24px',
    border: `1px solid ${border}`, textAlign: 'left',
    backdropFilter: purpleMode ? 'blur(20px)' : 'none',
    WebkitBackdropFilter: purpleMode ? 'blur(20px)' : 'none',
    boxShadow: purpleMode
      ? `0 0 0 1px ${accent}11, 0 32px 80px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08)`
      : darkMode ? '0 8px 40px rgba(0,0,0,0.4)' : '0 4px 24px rgba(0,0,0,0.06)',
  }
  const labelStyle = {
    fontSize: '0.78rem', color: subtext, textTransform: 'uppercase',
    letterSpacing: '0.06em', marginBottom: '0.4rem', display: 'block',
  }

  const handleSignup = async (e) => {
    e.preventDefault()
    if (formData.password !== formData.confirmPassword) return toast('Passwords do not match.', 'error')
    setLoading(true)
    try {
      const cred = await createUserWithEmailAndPassword(auth, formData.email, formData.password)
      await setDoc(doc(db, 'users', cred.user.uid), {
        uid: cred.user.uid, name: formData.name, email: formData.email,
        role, dob: formData.dob || null,
        verificationStatus: role === 'doctor' ? 'Pending (Local Verification)' : 'Verified',
        createdAt: new Date().toISOString(), provider: 'email',
      })
      await refreshUserData?.()
      navigate(role === 'doctor' ? '/dashboard/doctor' : '/dashboard/patient')
    } catch (err) {
      if (err.code === 'auth/email-already-in-use') {
        toast('This email is already registered. Please sign in instead.', 'error')
      } else if (err.code === 'auth/weak-password') {
        toast('Password must be at least 6 characters.', 'error')
      } else if (err.code === 'auth/invalid-email') {
        toast('Please enter a valid email address.', 'error')
      } else {
        toast(err.message.replace('Firebase: ', '').replace(/\s*\(auth\/.*?\)/, ''), 'error')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleConfirm = async () => {
    setShowGoogleDialog(false)
    setGoogleLoading(true)
    try {
      const provider = new GoogleAuthProvider()
      provider.setCustomParameters({ prompt: 'select_account' })
      const cred = await signInWithPopup(auth, provider)
      const snap = await getDoc(doc(db, 'users', cred.user.uid))
      if (snap.exists()) {
        navigate(snap.data().role === 'doctor' ? '/dashboard/doctor' : '/dashboard/patient')
        return
      }
      setGoogleUser(cred.user)
      setFormData(prev => ({ ...prev, name: cred.user.displayName || '', email: cred.user.email || '' }))
      setStep('google-profile')
    } catch (err) {
      if (err.code === 'auth/popup-closed-by-user') toast('Sign-in cancelled.', 'warning')
      else if (err.code === 'auth/popup-blocked') toast('Popup blocked — please allow popups for this site.', 'error')
      else if (err.code === 'auth/unauthorized-domain') toast('Domain not authorized. Add it in Firebase Console → Auth → Settings.', 'error')
      else { toast(`Google sign-in failed: ${err.code}`, 'error'); console.error(err) }
    } finally {
      setGoogleLoading(false)
    }
  }

  const handleGoogleProfileComplete = async (e) => {
    e.preventDefault()
    if (!role) return toast('Please select your role.', 'warning')
    setLoading(true)
    try {
      await setDoc(doc(db, 'users', googleUser.uid), {
        uid: googleUser.uid, name: formData.name, email: googleUser.email,
        role, dob: formData.dob || null,
        verificationStatus: role === 'doctor' ? 'Pending (Local Verification)' : 'Verified',
        createdAt: new Date().toISOString(), provider: 'google',
        photoURL: googleUser.photoURL || null,
      })
      await refreshUserData?.()
      const snap = await getDoc(doc(db, 'users', googleUser.uid))
      const savedRole = snap.data()?.role
      const destination = savedRole === 'doctor' ? '/dashboard/doctor' : '/dashboard/patient'
      navigate(destination)
    } catch (err) {
      toast('Failed to save profile. Please try again.', 'error')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const roles = [
    {
      id: 'doctor', label: 'Clinical User', sub: 'Doctor / Specialist',
      icon: <DoctorIcon size={26} color={accent}/>
    },
    {
      id: 'patient', label: 'Patient', sub: 'Personal Screening',
      icon: <PatientIcon size={26} color={accent}/>
    },
  ]

  return (
    <div style={{ backgroundColor: bg, color: text, minHeight: '100vh', fontFamily: 'Bricolage Grotesque, sans-serif', transition: 'background-color 0.4s, color 0.4s', overflowX: 'hidden' }}>
      <style>{`
        @keyframes dynamicReveal { 0%{opacity:0;transform:translateY(24px);filter:blur(8px)} 100%{opacity:1;transform:translateY(0);filter:blur(0px)} }
        @keyframes dialogIn      { 0%{opacity:0;transform:scale(0.92) translateY(16px)} 100%{opacity:1;transform:scale(1) translateY(0)} }
        @keyframes marquee       { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
        @keyframes pulse         { 0%,100%{opacity:0.6} 50%{opacity:1} }
        .stagger-1{opacity:0;animation:dynamicReveal 0.8s ease 0.1s forwards}
        .stagger-2{opacity:0;animation:dynamicReveal 0.8s ease 0.2s forwards}
        .stagger-3{opacity:0;animation:dynamicReveal 0.8s ease 0.3s forwards}
        .stagger-4{opacity:0;animation:dynamicReveal 0.8s ease 0.4s forwards}
        .stagger-5{opacity:0;animation:dynamicReveal 0.8s ease 0.5s forwards}
        .stagger-6{opacity:0;animation:dynamicReveal 0.8s ease 0.6s forwards}
        .stagger-7{opacity:0;animation:dynamicReveal 0.8s ease 0.7s forwards}
        .marquee-track{display:flex;width:max-content;animation:marquee 20s linear infinite}
        .marquee-track:hover{animation-play-state:paused}
        .role-card{transition:all 0.2s;cursor:pointer}
        .role-card:hover{transform:translateY(-4px) !important;border-color:${accent} !important}
        .btn-primary:hover{transform:translateY(-2px);box-shadow:0 0 32px ${accent}66 !important}
        .btn-secondary:hover{opacity:0.8}
        .google-btn:hover{opacity:0.9;transform:translateY(-1px)}
        .logo-btn{cursor:pointer;transition:opacity 0.2s}
        .logo-btn:hover{opacity:0.8}
        input::placeholder{color:${subtext};opacity:1}
        input[type="date"]::-webkit-calendar-picker-indicator{filter:${purpleMode || darkMode ? 'invert(1)' : 'none'};opacity:0.5;cursor:pointer}
      `}</style>

      {/* ── Navbar ── */}
      <nav style={{ ...navAnim, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.4rem 4rem', position: 'sticky', top: 0, backgroundColor: `${bg}ee`, backdropFilter: 'blur(12px)', zIndex: 100, borderBottom: `1px solid ${border}`, boxSizing: 'border-box' }}>
        <div className="logo-btn" onClick={() => navigate('/')} style={{ display: 'flex', alignItems: 'baseline' }}>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
          <span style={{ fontSize: '1.3rem', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
        </div>
        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
          <button onClick={() => setPurpleMode(p => !p)} title="Toggle violet theme"
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', opacity: purpleMode ? 1 : 0.5, fontSize: '1rem', transition: 'opacity 0.2s' }}
            onMouseEnter={e => e.currentTarget.style.opacity = 1}
            onMouseLeave={e => e.currentTarget.style.opacity = purpleMode ? 1 : 0.5}>✦</button>
          {!purpleMode && (
            <button onClick={() => setDarkMode(d => !d)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', opacity: 0.6, display: 'flex', transition: 'opacity 0.2s' }}
              onMouseEnter={e => e.currentTarget.style.opacity = 1} onMouseLeave={e => e.currentTarget.style.opacity = 0.6}>
              {darkMode
                ? <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
                : <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>}
            </button>
          )}
          <button onClick={() => navigate('/')} className="btn-secondary" style={btnSecondary}>Home</button>
          <button onClick={() => navigate('/login')} className="btn-primary" style={btnPrimary}>Sign In →</button>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', padding: '5rem 2rem 3rem', gap: '2rem' }}>
        <div style={{ ...badgeAnim, display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.35rem 1rem', borderRadius: '999px', border: `1px solid ${accent}44`, backgroundColor: `${accent}0d` }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: accent, display: 'inline-block', animation: 'pulse 1.5s ease-in-out infinite' }}/>
          <span style={{ fontSize: '0.78rem', color: accent, letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: '600' }}>AI Powered Medical Screening</span>
        </div>

        <h2 style={{ ...titleAnim, margin: '-0.5rem 0 0 0', display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: 'clamp(2.2rem, 5vw, 3.5rem)', fontWeight: '500', fontFamily: 'Bricolage Grotesque, sans-serif' }}>Join</span>
          <span style={{ fontSize: 'clamp(2.5rem, 6vw, 4rem)', fontWeight: '400', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
          <span style={{ fontSize: 'clamp(2.5rem, 6vw, 4rem)', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
        </h2>

        <p style={{ ...descAnim, fontSize: '1rem', color: subtext, maxWidth: '520px', lineHeight: 1.9, marginTop: '-0.5rem' }}>
          Empowering heart health through retinal intelligence.
        </p>

        <div style={{ ...formWrapperAnim, width: '100%', maxWidth: '480px' }}>

          {/* ── STEP: Role Selection ── */}
          {step === 'role' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
              <div style={{ display: 'flex', gap: '1rem' }}>
                {roles.map((r, i) => (
                  <div key={r.id} className={`stagger-${i + 1} role-card`}
                    onClick={() => { setRole(r.id); setStep('form') }}
                    style={{ flex: 1, padding: '2.2rem 1rem', borderRadius: '24px', ...cardStyle, textAlign: 'center', color: text }}>
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                      <div style={{ width: '52px', height: '52px', borderRadius: '14px', backgroundColor: `${accent}18`, border: `1px solid ${accent}33`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {r.icon}
                      </div>
                    </div>
                    <div style={{ fontWeight: '600', fontSize: '0.95rem' }}>{r.label}</div>
                    <div style={{ fontSize: '0.75rem', color: subtext, marginTop: '0.25rem' }}>{r.sub}</div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ flex: 1, height: '1px', backgroundColor: border }}/>
                <span style={{ fontSize: '0.75rem', color: subtext, whiteSpace: 'nowrap' }}>or continue with</span>
                <div style={{ flex: 1, height: '1px', backgroundColor: border }}/>
              </div>

              <button className="google-btn stagger-3" onClick={() => setShowGoogleDialog(true)} disabled={googleLoading}
                style={{ width: '100%', padding: '0.9rem', borderRadius: '14px', ...cardStyle, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', transition: 'all 0.2s', fontFamily: 'Bricolage Grotesque, sans-serif' }}>
                <svg width="20" height="20" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                <span style={{ fontSize: '0.92rem', fontWeight: '500', color: text }}>{googleLoading ? 'Opening Google...' : 'Continue with Google'}</span>
              </button>

              <p style={{ textAlign: 'center', fontSize: '0.82rem', color: subtext }}>
                Already have an account?{' '}
                <span onClick={() => navigate('/login')} style={{ color: accent, cursor: 'pointer', fontWeight: '500' }}>Sign in</span>
              </p>
            </div>
          )}

          {/* ── STEP: Email Form ── */}
          {step === 'form' && (
            <form onSubmit={handleSignup} style={cardStyle}>
              <button type="button" className="stagger-1" onClick={() => { setStep('role'); setRole(null) }}
                style={{ background: 'none', border: 'none', color: accent, cursor: 'pointer', marginBottom: '1.5rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '0.4rem', fontFamily: 'inherit', fontSize: '0.88rem' }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
                Back to roles
              </button>

              <div style={{ marginBottom: '1.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 0.9rem', borderRadius: '999px', backgroundColor: `${accent}18`, border: `1px solid ${accent}33` }}>
                <span style={{ fontSize: '0.75rem', color: accent, fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {role === 'doctor' ? 'Clinical User' : 'Patient'}
                </span>
              </div>

              <div className="stagger-2">
                <label style={labelStyle}>Full Name</label>
                <input type="text" placeholder="Your full name" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, name: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-3">
                <label style={labelStyle}>Email Address</label>
                <input type="email" placeholder="you@example.com" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, email: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-4">
                <label style={labelStyle}>Date of Birth <span style={{ color: subtext, fontSize: '0.7rem', textTransform: 'none', letterSpacing: 0 }}>(must be 18+)</span></label>
                <input type="date" max={maxDob} style={inputStyle} required
                  onChange={e => setFormData({ ...formData, dob: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-4">
                <label style={labelStyle}>Password</label>
                <input type="password" placeholder="Min. 8 characters" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, password: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-5">
                <label style={labelStyle}>Confirm Password</label>
                <input type="password" placeholder="Repeat password" style={inputStyle} required
                  onChange={e => setFormData({ ...formData, confirmPassword: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              {role === 'doctor' && (
                <div className="stagger-6" style={{ marginBottom: '1rem' }}>
                  <label style={labelStyle}>Medical License</label>
                  <div style={{ padding: '1rem', borderRadius: '12px', border: `1px dashed ${accent}44`, backgroundColor: `${accent}08`, display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.4rem' }}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    <input type="file" accept="image/*,.pdf" style={{ ...inputStyle, marginBottom: 0, padding: '0.3rem', border: 'none', backgroundColor: 'transparent', flex: 1 }}
                      onChange={e => setFormData({ ...formData, license: e.target.files[0] })} />
                  </div>
                  <p style={{ fontSize: '0.72rem', color: subtext, margin: 0 }}>JPG, PNG or PDF — for record-keeping only.</p>
                </div>
              )}

              <div className={role === 'doctor' ? 'stagger-7' : 'stagger-6'} style={{ marginTop: '0.5rem' }}>
                <button type="submit" disabled={loading} className="btn-primary"
                  style={{ ...btnPrimary, width: '100%', padding: '1rem', opacity: loading ? 0.7 : 1, cursor: loading ? 'not-allowed' : 'pointer' }}>
                  {loading ? 'Creating account...' : `Create ${role === 'doctor' ? 'Clinical' : 'Patient'} Account →`}
                </button>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '1.2rem 0 0.8rem' }}>
                <div style={{ flex: 1, height: '1px', backgroundColor: border }}/>
                <span style={{ fontSize: '0.72rem', color: subtext }}>or</span>
                <div style={{ flex: 1, height: '1px', backgroundColor: border }}/>
              </div>
              <button type="button" className="google-btn" onClick={() => setShowGoogleDialog(true)}
                style={{ width: '100%', padding: '0.8rem', borderRadius: '12px', backgroundColor: purpleMode ? 'rgba(255,255,255,0.04)' : darkMode ? '#0d0d0d' : '#f8f8f8', border: `1px solid ${border}`, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem', transition: 'all 0.2s', fontFamily: 'Bricolage Grotesque, sans-serif' }}>
                <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                <span style={{ fontSize: '0.88rem', fontWeight: '500', color: text }}>Continue with Google</span>
              </button>

              <p style={{ textAlign: 'center', marginTop: '1.2rem', fontSize: '0.85rem', color: subtext }}>
                Already have an account?{' '}
                <span onClick={() => navigate('/login')} style={{ color: accent, cursor: 'pointer', fontWeight: '500' }}>Sign in</span>
              </p>
            </form>
          )}

          {/* ── STEP: Google Profile Completion ── */}
          {step === 'google-profile' && (
            <form onSubmit={handleGoogleProfileComplete} style={cardStyle}>
              <div className="stagger-1" style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', padding: '0.9rem 1rem', borderRadius: '14px', backgroundColor: `${accent}0d`, border: `1px solid ${accent}22` }}>
                {googleUser?.photoURL
                  ? <img src={googleUser.photoURL} alt="avatar" style={{ width: '38px', height: '38px', borderRadius: '50%', border: `2px solid ${accent}` }} />
                  : <div style={{ width: '38px', height: '38px', borderRadius: '50%', backgroundColor: `${accent}33`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <PatientIcon size={18} color={subtext}/>
                    </div>
                }
                <div style={{ flex: 1, textAlign: 'left' }}>
                  <div style={{ fontWeight: '600', fontSize: '0.88rem', color: text }}>{googleUser?.displayName}</div>
                  <div style={{ fontSize: '0.75rem', color: subtext }}>{googleUser?.email}</div>
                </div>
                <div style={{ padding: '0.2rem 0.55rem', borderRadius: '999px', backgroundColor: '#34A85318', border: '1px solid #34A85333' }}>
                  <span style={{ fontSize: '0.65rem', color: '#34A853', fontWeight: '600' }}>Google</span>
                </div>
              </div>

              <p style={{ fontSize: '0.88rem', color: subtext, lineHeight: 1.6, margin: '0 0 1.5rem' }}>
                Almost there! Just a few more details to set up your account.
              </p>

              <div className="stagger-2">
                <label style={labelStyle}>Full Name</label>
                <input type="text" value={formData.name} style={inputStyle} required
                  onChange={e => setFormData({ ...formData, name: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              <div className="stagger-3" style={{ marginBottom: '1rem' }}>
                <label style={labelStyle}>I am a</label>
                <div style={{ display: 'flex', gap: '0.75rem' }}>
                  {[
                    { id: 'patient', label: 'Patient',       icon: <PatientIcon size={16} color={role === 'patient' ? accent : text}/> },
                    { id: 'doctor',  label: 'Clinical User', icon: <DoctorIcon  size={16} color={role === 'doctor'  ? accent : text}/> },
                  ].map(r => (
                    <button key={r.id} type="button" onClick={() => setRole(r.id)}
                      style={{ flex: 1, padding: '0.85rem', borderRadius: '14px', cursor: 'pointer', fontFamily: 'Bricolage Grotesque, sans-serif', fontWeight: '500', fontSize: '0.88rem', transition: 'all 0.2s', backgroundColor: role === r.id ? `${accent}22` : `${accent}08`, border: `1px solid ${role === r.id ? accent : border}`, color: role === r.id ? accent : text, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem' }}>
                      {r.icon}{r.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="stagger-4">
                <label style={labelStyle}>Date of Birth <span style={{ color: subtext, fontSize: '0.7rem', textTransform: 'none', letterSpacing: 0 }}>(must be 18+)</span></label>
                <input type="date" max={maxDob} style={inputStyle} required
                  onChange={e => setFormData({ ...formData, dob: e.target.value })}
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = border} />
              </div>

              {role === 'doctor' && (
                <div className="stagger-5" style={{ marginBottom: '1rem' }}>
                  <label style={labelStyle}>Medical License</label>
                  <div style={{ padding: '0.9rem', borderRadius: '12px', border: `1px dashed ${accent}44`, backgroundColor: `${accent}08`, display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.4rem' }}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    <input type="file" accept="image/*,.pdf" style={{ ...inputStyle, marginBottom: 0, padding: '0.3rem', border: 'none', backgroundColor: 'transparent', flex: 1 }}
                      onChange={e => setFormData({ ...formData, license: e.target.files[0] })} />
                  </div>
                  <p style={{ fontSize: '0.72rem', color: subtext, margin: 0 }}>JPG, PNG or PDF — for record-keeping only.</p>
                </div>
              )}

              <div className="stagger-5" style={{ marginTop: '0.5rem' }}>
                <button type="submit" disabled={loading || !role} className="btn-primary"
                  style={{ ...btnPrimary, width: '100%', padding: '1rem', opacity: (!role || loading) ? 0.6 : 1, cursor: (!role || loading) ? 'not-allowed' : 'pointer' }}>
                  {loading ? 'Saving profile...' : 'Complete Sign Up →'}
                </button>
              </div>
            </form>
          )}
        </div>
      </section>

      {/* ── Marquee ── */}
      <section style={{ ...techAnim, padding: '2rem 0 4rem', overflow: 'hidden' }}>
        <p style={{ textAlign: 'center', fontSize: '0.75rem', color: subtext, letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: '500', marginBottom: '2rem' }}>Built with</p>
        <div style={{ overflow: 'hidden', position: 'relative' }}>
          <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to right, ${bg}, transparent)`, zIndex: 2, pointerEvents: 'none' }}/>
          <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: '120px', background: `linear-gradient(to left, ${bg}, transparent)`, zIndex: 2, pointerEvents: 'none' }}/>
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

      <footer style={{ textAlign: 'center', padding: '2rem', borderTop: `1px solid ${border}`, color: subtext, fontSize: '0.82rem' }}>
        © 2026 CardioRetina — AI Powered Cardiovascular Screening
      </footer>

      <ToastContainer toasts={toasts} dismiss={dismiss} darkMode={darkMode} />
      {showGoogleDialog && (
        <GooglePermissionDialog
          onConfirm={handleGoogleConfirm} onCancel={() => setShowGoogleDialog(false)}
          accent={accent} bg={bg} text={text} subtext={subtext} border={border}
          purpleMode={purpleMode} darkMode={darkMode}
        />
      )}
    </div>
  )
}

export default Signup
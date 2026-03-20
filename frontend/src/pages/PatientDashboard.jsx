import { useState, useEffect } from 'react'
import { useToast, ToastContainer } from './Toast'
import { useNavigate } from 'react-router-dom'
import { auth, db } from '../firebase'
import { signOut } from 'firebase/auth'
import { collection, addDoc, getDocs, query, orderBy, where, serverTimestamp } from 'firebase/firestore'
import { useAuth } from '../context/AuthContext'

function useAnimate(delay = 0) {
  const [style, setStyle] = useState({
    opacity: 0, transform: 'translateY(24px)', filter: 'blur(8px)',
    transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms`
  })
  useEffect(() => {
    const t = setTimeout(() => {
      setStyle({
        opacity: 1, transform: 'translateY(0)', filter: 'blur(0px)',
        transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms, filter 0.8s ease ${delay}ms`
      })
    }, 50)
    return () => clearTimeout(t)
  }, [])
  return style
}

function RiskGauge({ score, label, darkMode }) {
  const [animated, setAnimated] = useState(0)
  
  const accent = darkMode ? '#d5ff5f' : '#bce236'
  
  useEffect(() => {
    const t = setTimeout(() => setAnimated(score), 300)
    return () => clearTimeout(t)
  }, [score])
  
  const isHighRisk = label === 'High Risk'
  const color = isHighRisk ? '#ff4d4d' : accent
  const radius = 80
  const stroke = 10
  const normalizedRadius = radius - stroke
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (animated / 100) * circumference
  
  return (
    <div style={{ position: 'relative', width: `${radius * 2}px`, height: `${radius * 2}px`, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      
      <svg height={radius * 2} width={radius * 2} style={{ position: 'absolute', top: 0, left: 0, transform: 'rotate(-90deg)' }}>
        <circle stroke={darkMode ? '#222' : '#eee'} fill="transparent" strokeWidth={stroke} r={normalizedRadius} cx={radius} cy={radius} />
        <circle stroke={color} fill="transparent" strokeWidth={stroke} strokeDasharray={`${circumference} ${circumference}`} strokeDashoffset={strokeDashoffset} strokeLinecap="round" r={normalizedRadius} cx={radius} cy={radius} style={{ transition: 'stroke-dashoffset 1.2s ease' }} />
      </svg>
      
      <div style={{ zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '10px' }}>
        <div style={{ fontSize: '1.8rem', fontWeight: '700', color, fontFamily: 'Junicode, serif', lineHeight: 1 }}>{score.toFixed(1)}%</div>
        <div style={{ marginTop: '0.4rem', padding: '0.2rem 0.6rem', borderRadius: '999px', backgroundColor: isHighRisk ? '#ff4d4d22' : `${accent}22`, color, fontSize: '0.75rem', fontWeight: '600', whiteSpace: 'nowrap' }}>{label}</div>
      </div>

    </div>
  )
}


function ConsultCard({ results, appointmentSent, selectedDoctor, requesting, onSelectDoctor, onRequestAppointment, onTabChange, accent, card, card2, bg, text, subtext, border, darkMode }) {
  const isHighRisk = results?.label === 'High Risk'
  const hasResults = !!results

  // Colors based on risk state
  const borderColor = isHighRisk ? '#ff4d4d33' : `${accent}33`
  const accentColor = isHighRisk ? '#ff4d4d' : accent
  const iconColor   = isHighRisk ? '#ff4d4d' : accent

  const titleText = isHighRisk
    ? 'High Risk Detected — '
    : hasResults
      ? 'Low Risk — '
      : 'Consult a '

  const titleAccent = isHighRisk ? 'Consult a Doctor' : hasResults ? 'Still want a Consultation?' : 'Doctor'

  const bodyText = isHighRisk
    ? 'Your scan indicates cardiovascular risk. We strongly recommend consulting a cardiologist.'
    : hasResults
      ? 'Your scan looks healthy. You can still book a doctor consultation for a professional review.'
      : 'You can request a doctor appointment anytime — even before running a scan.'

  return (
    <div style={{ marginTop: '2rem', backgroundColor: card, borderRadius: '24px', border: `1px solid ${borderColor}`, padding: '2rem' }}>
      {!appointmentSent ? (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
            {isHighRisk ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={iconColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={iconColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>
            )}
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
              {titleText}<span style={{ color: accent, fontStyle: 'italic' }}>{titleAccent}</span>
            </h3>
          </div>

          <p style={{ margin: '0 0 1.5rem', color: subtext, fontSize: '0.9rem', lineHeight: 1.6 }}>{bodyText}</p>

          {/* No scan badge */}
          {!hasResults && (
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.3rem 0.8rem', borderRadius: '999px', backgroundColor: `${accent}11`, border: `1px solid ${accent}33`, marginBottom: '1.2rem' }}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <span style={{ fontSize: '0.75rem', color: accent }}>No scan attached — doctor will be notified</span>
            </div>
          )}

          {/* Selected doctor preview */}
          {selectedDoctor && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.4rem', borderRadius: '16px', border: `1.5px solid ${accent}`, backgroundColor: darkMode ? '#1a2600' : '#f6ffe6', marginBottom: '1.2rem' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '600', fontSize: '0.95rem', marginBottom: '0.2rem' }}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>
                  {selectedDoctor.name}
                </div>
                <div style={{ color: subtext, fontSize: '0.82rem' }}>{selectedDoctor.specialty || 'Cardiologist'} · {selectedDoctor.email}</div>
              </div>
              <button onClick={onSelectDoctor} style={{ background: 'none', border: 'none', color: subtext, fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'inherit', textDecoration: 'underline' }}>Change</button>
            </div>
          )}

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            {!selectedDoctor && (
              <button onClick={onSelectDoctor}
                style={{ padding: '0.8rem 2rem', borderRadius: '999px', backgroundColor: 'transparent', color: accentColor, fontWeight: '600', border: `1px solid ${accentColor}`, cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'all 0.2s' }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>
                Select a Doctor →
              </button>
            )}
            {selectedDoctor && (
              <button onClick={onRequestAppointment} disabled={requesting}
                style={{ padding: '0.8rem 2rem', borderRadius: '999px', backgroundColor: requesting ? card2 : accentColor, color: isHighRisk ? '#fff' : '#000', fontWeight: '600', border: 'none', cursor: requesting ? 'not-allowed' : 'pointer', fontSize: '0.95rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: requesting ? 0.7 : 1, transition: 'all 0.2s' }}>
                {requesting
                  ? <><span style={{ width: '16px', height: '16px', border: `2px solid ${isHighRisk ? '#fff' : '#000'}`, borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.8s linear infinite' }}/>Sending...</>
                  : `Request Appointment with ${selectedDoctor.name} →`}
              </button>
            )}
          </div>
        </>
      ) : (
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ width: '44px', height: '44px', borderRadius: '50%', backgroundColor: `${accent}22`, border: `1.5px solid ${accent}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
          </div>
          <div>
            <div style={{ fontWeight: '600', fontSize: '0.95rem', color: accent, marginBottom: '0.2rem' }}>Appointment request sent!</div>
            <div style={{ color: subtext, fontSize: '0.85rem' }}>
              Request sent to <strong style={{ color: text }}>{selectedDoctor?.name}</strong>. Check your{' '}
              <span style={{ color: accent, cursor: 'pointer', textDecoration: 'underline' }} onClick={onTabChange}>Appointments tab</span> for updates.
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function PatientDashboard({ darkMode, setDarkMode }) {
  const navigate = useNavigate()
  const { currentUser, userData } = useAuth()

  const [activeTab, setActiveTab] = useState('My Scan')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [scanHistory, setScanHistory] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(true)

  // ── Doctor selection + appointments ──────────────────────
  const [doctors, setDoctors] = useState([])
  const { toasts, toast, dismiss } = useToast()
  const [selectedDoctor, setSelectedDoctor] = useState(null)
  const [showDoctorModal, setShowDoctorModal] = useState(false)
  const [appointmentSent, setAppointmentSent] = useState(false)
  const [requesting, setRequesting] = useState(false)
  const [appointments, setAppointments] = useState([])
  const [loadingApts, setLoadingApts] = useState(true)

  const accent = darkMode ? '#d5ff5f' : '#bce236'
  const bg = darkMode ? '#000000' : '#ffffff'
  const card = darkMode ? '#111111' : '#f5f5f5'
  const card2 = darkMode ? '#1a1a1a' : '#eeeeee'
  const text = darkMode ? '#ffffff' : '#111111'
  const subtext = darkMode ? '#9ca3af' : '#6b7280'
  const border = darkMode ? '#222222' : '#e5e5e5'

  const navAnim = useAnimate(0)
  const contentAnim = useAnimate(300)
  const resultAnim = useAnimate(200)

  useEffect(() => {
    if (!currentUser) return
    const fetchHistory = async () => {
      try {
        const q = query(
          collection(db, 'scans', currentUser.uid, 'history'),
          orderBy('timestamp', 'desc')
        )
        const snapshot = await getDocs(q)
        const history = snapshot.docs.map(doc => doc.data())
        setScanHistory(history)
      } catch (err) {
        console.error('Error fetching history:', err)
      } finally {
        setLoadingHistory(false)
      }
    }
    const fetchDoctors = async () => {
      try {
        const q = query(collection(db, 'users'), where('role', '==', 'doctor'))
        const snap = await getDocs(q)
        setDoctors(snap.docs.map(d => ({ id: d.id, ...d.data() })))
      } catch (err) {
        console.error('Error fetching doctors:', err)
      }
    }
    const fetchAppointments = async () => {
      try {
        const q = query(
          collection(db, 'appointments'),
          where('patientId', '==', currentUser.uid),
          orderBy('createdAt', 'desc')
        )
        const snap = await getDocs(q)
        setAppointments(snap.docs.map(d => ({ id: d.id, ...d.data() })))
      } catch (err) {
        console.error('Error fetching appointments:', err)
      } finally {
        setLoadingApts(false)
      }
    }
    fetchHistory()
    fetchDoctors()
    fetchAppointments()
  }, [currentUser])

  const handleLogout = async () => {
    await signOut(auth)
    navigate('/login')
  }

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    if (selected) {
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResults(null)
      setSelectedDoctor(null)
      setAppointmentSent(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) {
      setFile(dropped)
      setPreview(URL.createObjectURL(dropped))
      setResults(null)
      setSelectedDoctor(null)
      setAppointmentSent(false)
    }
  }

  const runAnalysis = async () => {
    if (!file) return
    setIsAnalyzing(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (data.error) {
        toast(data.error, 'error')
        setFile(null)
        setPreview(null)
        setIsAnalyzing(false)
        return
      }

      const now = new Date()
      const dateStr = now.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })

      const scanEntry = {
        date: dateStr,
        label: data.label,
        score: data.risk_score,
        timestamp: now.getTime()
      }

      try {
        await addDoc(collection(db, 'scans', currentUser.uid, 'history'), scanEntry)
        setScanHistory(prev => [scanEntry, ...prev])
      } catch (err) {
        console.error('Error saving scan:', err)
      }

      setResults({
        risk_score:     data.risk_score,
        label:          data.label,
        confidence:     data.confidence,
        conditions:     data.conditions,
        recommendation: data.recommendation,
        gradcam:        data.gradcam,
        image_b64:      data.image_b64  || null,
      })

    } catch (err) {
      console.error('Failed to connect to backend:', err)
      toast('Failed to connect to the AI server. Please try again.', 'error')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const requestAppointment = async () => {
    if (!selectedDoctor) return
    setRequesting(true)
    try {
      await addDoc(collection(db, 'appointments'), {
        patientId: currentUser.uid,
        patientName: userData?.name || currentUser.email,
        doctorId: selectedDoctor.id,
        doctorName: selectedDoctor.name,
        scanResult: results ? {
          riskScore:      results.risk_score,
          label:          results.label,
          conditions:     results.conditions,
          recommendation: results.recommendation,
          imageb64:       results.image_b64  || null,
          gradcam:        results.gradcam    || null,
        } : null,
        status: 'pending',
        appointmentTime: null,
        createdAt: serverTimestamp(),
      })
      setAppointmentSent(true)
      // Refresh appointments list
      const q = query(collection(db, 'appointments'), where('patientId', '==', currentUser.uid), orderBy('createdAt', 'desc'))
      const snap = await getDocs(q)
      setAppointments(snap.docs.map(d => ({ id: d.id, ...d.data() })))
    } catch (err) {
      toast('Error sending request: ' + err.message, 'error')
    } finally {
      setRequesting(false)
    }
  }

  const handleDownloadReport = () => {
    if (!results) return;

    const printWindow = window.open('', '_blank');
    const dateStr = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const patientName = userData?.name || currentUser?.email || 'Patient';
    
    const printAccent = '#88b500'; 
    const riskColor = results.label === 'High Risk' ? '#ff4d4d' : printAccent;

    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <title>CardioRetina Report - ${patientName}</title>
          <style>
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #111; padding: 40px; line-height: 1.6; }
            .header { border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-end; }
            .title { font-size: 28px; margin: 0; font-weight: bold; }
            .subtitle { color: #666; font-size: 14px; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
            .patient-info { margin-bottom: 40px; display: flex; justify-content: space-between; background: #f9f9f9; padding: 20px; border-radius: 8px; }
            .info-block { display: flex; flex-direction: column; }
            .info-label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
            .info-value { font-size: 16px; font-weight: bold; }
            .results-container { display: flex; gap: 20px; margin-bottom: 30px; }
            .box { border: 1px solid #e5e5e5; padding: 24px; border-radius: 12px; flex: 1; }
            .box-title { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 15px; }
            .score { font-size: 48px; font-weight: bold; margin: 10px 0; color: ${riskColor}; }
            .badge { display: inline-block; padding: 6px 16px; border-radius: 999px; background-color: ${results.label === 'High Risk' ? '#ff4d4d22' : '#88b50022'}; color: ${riskColor}; font-weight: bold; font-size: 14px; }
            ul { margin: 0; padding-left: 20px; }
            li { margin-bottom: 8px; }
            .images { display: flex; gap: 20px; margin-bottom: 30px; page-break-inside: avoid; }
            .img-box { flex: 1; }
            img { width: 100%; max-width: 300px; border-radius: 8px; border: 1px solid #eee; display: block; margin-top: 10px; }
            .footer { text-align: center; font-size: 11px; color: #888; margin-top: 50px; border-top: 1px solid #eee; padding-top: 20px; page-break-inside: avoid; }
          </style>
        </head>
        <body>
          <div class="header">
            <div>
              <h1 class="title">CardioRetina</h1>
              <div class="subtitle">Cardiovascular Risk Screening Report</div>
            </div>
            <div style="text-align: right; color: #666; font-size: 14px;">
              Generated on<br/><strong>${dateStr}</strong>
            </div>
          </div>
          <div class="patient-info">
            <div class="info-block"><span class="info-label">Patient Name</span><span class="info-value">${patientName}</span></div>
            <div class="info-block"><span class="info-label">Report ID</span><span class="info-value">CR-${Math.random().toString(36).substr(2, 9).toUpperCase()}</span></div>
            <div class="info-block"><span class="info-label">Scan Type</span><span class="info-value">Retinal Fundus Image</span></div>
          </div>
          <div class="results-container">
            <div class="box" style="flex: 0.7; text-align: center;">
              <div class="box-title">Risk Assessment</div>
              <div class="score">${results.risk_score.toFixed(1)}%</div>
              <div class="badge">${results.label}</div>
            </div>
            <div class="box" style="flex: 1.3;">
              <div class="box-title">Detected Biomarkers</div>
              <ul>${results.conditions.map(c => `<li><strong>${c}</strong></li>`).join('')}</ul>
            </div>
          </div>
          <div class="images">
            <div class="box img-box"><div class="box-title">Original Scan</div><img src="${preview}" /></div>
            <div class="box img-box"><div class="box-title">Grad-CAM Heatmap Analysis</div>${results.gradcam ? `<img src="data:image/jpeg;base64,${results.gradcam}" />` : 'No heatmap generated'}</div>
          </div>
          <div class="box"><div class="box-title">Clinical Recommendation</div><p style="margin: 0; font-size: 15px;">${results.recommendation}</p></div>
          <div class="footer">Disclaimer: This report was generated by an AI screening prototype (CardioRetina) and is intended for informational and educational purposes only. It does not replace a formal medical diagnosis. Please consult with a qualified healthcare professional or cardiologist for clinical advice and proper medical evaluation.</div>
        </body>
      </html>
    `;
    
    printWindow.document.write(html);
    printWindow.document.close();
    setTimeout(() => { printWindow.focus(); printWindow.print(); }, 500);
  }

  const lastScan = scanHistory[0] || null
  const tabs = ['My Scan', 'Appointments', 'History', 'Profile']

  return (
    <div style={{ minHeight: '100vh', backgroundColor: bg, color: text, fontFamily: 'Bricolage Grotesque, sans-serif', transition: 'background-color 0.3s ease' }}>

      <nav style={{ ...navAnim, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.2rem 3rem', borderBottom: `1px solid ${border}`, position: 'sticky', top: 0, backgroundColor: bg, zIndex: 100 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', cursor: 'pointer' }} onClick={() => navigate('/')}>
          <span style={{ fontSize: '1.2rem', fontWeight: '400', fontFamily: 'Junicode, serif', color: text }}>Cardio</span>
          <span style={{ fontSize: '1.2rem', fontWeight: '400', fontStyle: 'italic', fontFamily: 'Junicode, serif', color: accent }}>Retina</span>
        </div>

        <div style={{ display: 'flex', gap: '0.3rem', backgroundColor: card, padding: '0.3rem', borderRadius: '999px' }}>
          {tabs.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              padding: '0.5rem 1.3rem', borderRadius: '999px', border: 'none',
              backgroundColor: activeTab === tab ? (darkMode ? '#ffffff' : '#111111') : 'transparent',
              color: activeTab === tab ? (darkMode ? '#000000' : '#ffffff') : subtext,
              fontWeight: activeTab === tab ? '600' : '500',
              fontSize: '0.9rem', fontFamily: 'inherit', cursor: 'pointer', transition: 'all 0.2s',
              position: 'relative'
            }}>
              {tab}
              {tab === 'Appointments' && appointments.filter(a => a.status === 'pending').length > 0 && (
                <span style={{ position: 'absolute', top: '4px', right: '6px', width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#f59e0b' }} />
              )}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
          <div style={{ fontSize: '0.85rem', color: subtext }}>{userData?.name || currentUser?.email || 'Patient'}</div>
          <button onClick={() => setDarkMode(!darkMode)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', display: 'flex', alignItems: 'center', opacity: 0.7, transition: 'opacity 0.2s' }} onMouseEnter={e => e.currentTarget.style.opacity = 1} onMouseLeave={e => e.currentTarget.style.opacity = 0.7}>
            {darkMode ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
            )}
          </button>
          <button onClick={handleLogout} style={{ padding: '0.5rem 1.2rem', borderRadius: '999px', backgroundColor: 'transparent', color: '#ff4d4d', border: '1px solid #ff4d4d33', cursor: 'pointer', fontSize: '0.85rem', fontFamily: 'inherit', fontWeight: '500' }}>Sign Out</button>
        </div>
      </nav>

      <main style={{ maxWidth: '1100px', margin: '0 auto', padding: '3rem 2rem' }}>

        {activeTab === 'My Scan' && (
          <div style={contentAnim}>
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>My <span style={{ color: accent, fontStyle: 'italic' }}>Scan</span></h1>
              <p style={{ margin: '0.4rem 0 0', color: subtext, fontSize: '0.95rem' }}>Upload your retinal fundus image for cardiovascular risk screening.</p>
            </div>

            {!results && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
                {[
                  { label: 'Last Scan', value: lastScan ? lastScan.date.split(',')[0] : 'No scans yet', sub: lastScan ? `${lastScan.score}% — ${lastScan.label}` : 'Upload your first scan', color: lastScan ? accent : subtext },
                  { label: 'Total Scans', value: String(scanHistory.length), sub: scanHistory.length === 0 ? 'No scans yet' : `${scanHistory.length} scan${scanHistory.length > 1 ? 's' : ''} completed`, color: text },
                  { label: 'Overall Status', value: lastScan ? lastScan.label : 'Unknown', sub: lastScan ? 'Based on latest scan' : 'Run a scan to see status', color: lastScan?.label === 'High Risk' ? '#ff4d4d' : lastScan ? accent : subtext },
                ].map((item, i) => (
                  <div key={i} style={{ backgroundColor: card, padding: '1.5rem', borderRadius: '20px', border: `1px solid ${border}` }}>
                    <p style={{ margin: '0 0 0.5rem', fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{item.label}</p>
                    <p style={{ margin: '0 0 0.2rem', fontSize: '1.5rem', fontWeight: '700', color: item.color, fontFamily: 'Junicode, serif' }}>{item.value}</p>
                    <p style={{ margin: 0, fontSize: '0.8rem', color: subtext }}>{item.sub}</p>
                  </div>
                ))}
              </div>
            )}

            {!preview && (
              <div onDragOver={(e) => { e.preventDefault(); setDragOver(true) }} onDragLeave={() => setDragOver(false)} onDrop={handleDrop} style={{ border: `2px dashed ${dragOver ? accent : border}`, borderRadius: '24px', padding: '4rem 2rem', textAlign: 'center', backgroundColor: dragOver ? (darkMode ? '#111' : '#f9f9f9') : card, transition: 'all 0.2s', cursor: 'pointer', position: 'relative' }}>
                <input type="file" accept="image/*" onChange={handleFileChange} style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%', height: '100%' }} />
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={dragOver ? accent : subtext} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '1rem' }}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <h3 style={{ margin: '0 0 0.4rem', fontSize: '1.1rem', fontWeight: '500', color: dragOver ? accent : text }}>{dragOver ? 'Drop to upload' : 'Upload Your Retinal Scan'}</h3>
                <p style={{ margin: 0, color: subtext, fontSize: '0.875rem' }}>Drag and drop or click to browse — JPEG, PNG supported</p>
              </div>
            )}

            {preview && !results && (
              <div style={{ marginTop: '2rem', display: 'flex', gap: '2rem', alignItems: 'center', backgroundColor: card, padding: '2rem', borderRadius: '24px', border: `1px solid ${border}` }}>
                <img src={preview} alt="Preview" style={{ width: '160px', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '16px', border: `1px solid ${border}` }} />
                <div style={{ flex: 1 }}>
                  <h4 style={{ margin: '0 0 0.5rem', fontSize: '1.1rem', fontWeight: '600' }}>Image Ready</h4>
                  <p style={{ color: subtext, fontSize: '0.9rem', marginBottom: '1.5rem', lineHeight: 1.6 }}>Your retinal scan will be analyzed for cardiovascular risk biomarkers using our AI model.</p>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <button onClick={runAnalysis} disabled={isAnalyzing} style={{ padding: '0.8rem 2rem', borderRadius: '999px', backgroundColor: isAnalyzing ? card2 : accent, color: '#000', fontWeight: '600', border: 'none', cursor: isAnalyzing ? 'not-allowed' : 'pointer', fontSize: '0.95rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: isAnalyzing ? 0.7 : 1, transition: 'all 0.2s' }}>
                      {isAnalyzing ? (<><span style={{ width: '16px', height: '16px', border: '2px solid #000', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.8s linear infinite' }}/>Analyzing...</>) : 'Analyze My Scan →'}
                    </button>
                    <button onClick={() => { setFile(null); setPreview(null) }} style={{ padding: '0.8rem 1.5rem', borderRadius: '999px', backgroundColor: 'transparent', color: subtext, border: `1px solid ${border}`, cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit' }}>Remove</button>
                  </div>
                </div>
              </div>
            )}

            {/* ── Consult card — visible before scan ── */}
            {!results && !preview && (
              <ConsultCard
                results={null}
                appointmentSent={appointmentSent}
                selectedDoctor={selectedDoctor}
                requesting={requesting}
                onSelectDoctor={() => setShowDoctorModal(true)}
                onRequestAppointment={requestAppointment}
                onTabChange={() => setActiveTab('Appointments')}
                accent={accent} card={card} card2={card2} bg={bg}
                text={text} subtext={subtext} border={border} darkMode={darkMode}
              />
            )}

            {results && !isAnalyzing && (
              <div style={{ ...resultAnim, marginTop: '2.5rem' }}>
                <h2 style={{ margin: '0 0 2rem', fontSize: '1.6rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>Your <span style={{ color: accent, fontStyle: 'italic' }}>Results</span></h2>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: '1.5rem' }}>
                  <div style={{ backgroundColor: card, padding: '2rem', borderRadius: '24px', border: `1px solid ${border}`, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1.5rem' }}>
                    <p style={{ margin: 0, fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Risk Score</p>
                    <RiskGauge score={results.risk_score} label={results.label} darkMode={darkMode} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div style={{ backgroundColor: card, padding: '1.5rem', borderRadius: '24px', border: `1px solid ${border}` }}>
                      <p style={{ margin: '0 0 1rem', fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Heatmap Analysis</p>
                      <div style={{ display: 'flex', gap: '1rem' }}>
                        <div style={{ flex: 1 }}><p style={{ fontSize: '0.75rem', color: subtext, margin: '0 0 0.5rem' }}>Original</p><img src={preview} style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '12px' }} alt="Original" /></div>
                        <div style={{ flex: 1 }}><p style={{ fontSize: '0.75rem', color: accent, margin: '0 0 0.5rem' }}>Grad-CAM</p>
                          {results?.gradcam ? (<img src={`data:image/jpeg;base64,${results.gradcam}`} style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '12px' }} alt="Grad-CAM" />) : (<div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '12px', background: `linear-gradient(135deg, ${accent}22, ${accent}55)`, border: `1px solid ${accent}44`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span style={{ fontSize: '0.75rem', color: accent }}>No heatmap</span></div>)}
                        </div>
                      </div>
                    </div>
                    <div style={{ backgroundColor: card, padding: '1.5rem', borderRadius: '24px', border: `1px solid ${border}` }}>
                      <p style={{ margin: '0 0 0.8rem', fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Findings</p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.5rem' }}>
                        {results.conditions.map((c, i) => (<span key={i} style={{ padding: '0.3rem 0.8rem', borderRadius: '999px', backgroundColor: results.label === 'High Risk' ? '#ff4d4d22' : `${accent}22`, color: results.label === 'High Risk' ? '#ff4d4d' : accent, fontSize: '0.8rem', fontWeight: '500' }}>{c}</span>))}
                      </div>
                      <p style={{ margin: '0 0 0.8rem', fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Doctor's Note</p>
                      <p style={{ margin: 0, fontSize: '0.9rem', color: text, lineHeight: 1.7, padding: '1rem', borderRadius: '12px', backgroundColor: bg, border: `1px solid ${border}` }}>{results.recommendation}</p>
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                  <button onClick={handleDownloadReport} style={{ padding: '0.8rem 2rem', borderRadius: '999px', backgroundColor: accent, color: '#000', fontWeight: '600', border: 'none', cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit' }}>Download PDF Report →</button>
                  <button onClick={() => { setFile(null); setPreview(null); setResults(null); setSelectedDoctor(null); setAppointmentSent(false) }} style={{ padding: '0.8rem 2rem', borderRadius: '999px', backgroundColor: card, color: text, fontWeight: '500', border: `1px solid ${border}`, cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit' }}>New Scan</button>
                </div>

                {/* ── Consult a Doctor — always visible after results ── */}
                <ConsultCard
                  results={results}
                  appointmentSent={appointmentSent}
                  selectedDoctor={selectedDoctor}
                  requesting={requesting}
                  onSelectDoctor={() => setShowDoctorModal(true)}
                  onRequestAppointment={requestAppointment}
                  onTabChange={() => setActiveTab('Appointments')}
                  accent={accent} card={card} card2={card2} bg={bg}
                  text={text} subtext={subtext} border={border} darkMode={darkMode}
                />
              </div>
            )}
          </div>
        )}

        {activeTab === 'Appointments' && (
          <div style={contentAnim}>
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
                My <span style={{ color: accent, fontStyle: 'italic' }}>Appointments</span>
              </h1>
              <p style={{ margin: '0.4rem 0 0', color: subtext, fontSize: '0.95rem' }}>Track your appointment requests and confirmed schedules.</p>
            </div>
            {loadingApts ? (
              <div style={{ textAlign: 'center', padding: '3rem', color: subtext }}>Loading appointments...</div>
            ) : appointments.length === 0 ? (
              <div style={{ backgroundColor: card, borderRadius: '24px', border: `1px solid ${border}`, padding: '3rem', textAlign: 'center', color: subtext }}>
                No appointments yet. Run a scan and select a doctor if you're at high risk.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {appointments.map((apt) => {
                  const isHigh = apt.scanResult?.label === 'High Risk'
                  const statusColor = apt.status === 'scheduled' ? accent : apt.status === 'pending' ? '#f59e0b' : subtext
                  return (
                    <div key={apt.id} style={{ backgroundColor: card, padding: '1.8rem 2rem', borderRadius: '24px', border: `1px solid ${border}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '600', fontSize: '1rem', marginBottom: '0.4rem' }}>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={subtext} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>
                            {apt.doctorName}
                          </div>
                          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                            <span style={{ padding: '0.25rem 0.75rem', borderRadius: '999px', backgroundColor: `${statusColor}22`, color: statusColor, fontSize: '0.78rem', fontWeight: '600', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                              {apt.status === 'pending' && <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>Pending</>}
                              {apt.status === 'scheduled' && <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>Scheduled</>}
                              {apt.status === 'done'     && <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>Completed</>}
                              {apt.status === 'rejected' && <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>Rejected</>}
                            </span>
                            <span style={{ padding: '0.25rem 0.75rem', borderRadius: '999px', backgroundColor: isHigh ? '#ff4d4d22' : `${accent}22`, color: isHigh ? '#ff4d4d' : accent, fontSize: '0.78rem', fontWeight: '600' }}>
                              {apt.scanResult?.label} — {apt.scanResult?.riskScore}%
                            </span>
                          </div>
                        </div>
                        {apt.appointmentTime && (
                          <div style={{ textAlign: 'right' }}>
                            <p style={{ margin: '0 0 0.2rem', fontSize: '0.75rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Appointment Time</p>
                            <p style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700', color: accent, fontFamily: 'Junicode, serif' }}>
                              {new Date(apt.appointmentTime).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
                            </p>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: subtext }}>
                              {new Date(apt.appointmentTime).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
                            </p>
                          </div>
                        )}
                      </div>
                      {apt.scanResult?.conditions?.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.5rem' }}>
                          {apt.scanResult.conditions.map((c, i) => (
                            <span key={i} style={{ padding: '0.2rem 0.65rem', borderRadius: '999px', backgroundColor: isHigh ? '#ff4d4d11' : `${accent}11`, color: isHigh ? '#ff4d4d' : accent, fontSize: '0.75rem', border: `1px solid ${isHigh ? '#ff4d4d33' : `${accent}33`}` }}>{c}</span>
                          ))}
                        </div>
                      )}
                      {apt.status === 'pending' && (
                        <p style={{ margin: '1rem 0 0', fontSize: '0.85rem', color: subtext, padding: '0.8rem 1rem', borderRadius: '12px', backgroundColor: bg, border: `1px solid ${border}` }}>
                          Your request is with Dr. {apt.doctorName}. You'll see the appointment time here once they confirm.
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {activeTab === 'History' && (
          <div style={contentAnim}>
            <h1 style={{ margin: '0 0 0.5rem', fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>Scan <span style={{ color: accent, fontStyle: 'italic' }}>History</span></h1>
            <p style={{ color: subtext, marginBottom: '2rem' }}>Your past retinal scans and results.</p>
            {loadingHistory ? (<div style={{ textAlign: 'center', padding: '3rem', color: subtext }}>Loading history...</div>) : scanHistory.length === 0 ? (<div style={{ backgroundColor: card, borderRadius: '24px', border: `1px solid ${border}`, padding: '3rem', textAlign: 'center', color: subtext }}>No scans yet. Run your first analysis to see history here.</div>) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {scanHistory.map((scan, i) => (
                  <div key={i} style={{ backgroundColor: card, padding: '1.5rem 2rem', borderRadius: '20px', border: `1px solid ${border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div><p style={{ margin: '0 0 0.3rem', fontWeight: '600' }}>{scan.date}</p><p style={{ margin: 0, fontSize: '0.85rem', color: subtext }}>Cardiovascular Risk Scan</p></div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <span style={{ fontSize: '1.2rem', fontWeight: '700', color: scan.label === 'High Risk' ? '#ff4d4d' : accent, fontFamily: 'Junicode, serif' }}>{scan.score}%</span>
                      <span style={{ padding: '0.3rem 0.8rem', borderRadius: '999px', backgroundColor: scan.label === 'High Risk' ? '#ff4d4d22' : `${accent}22`, color: scan.label === 'High Risk' ? '#ff4d4d' : accent, fontSize: '0.8rem', fontWeight: '600' }}>{scan.label}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'Profile' && (
          <div style={{ ...contentAnim, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: '100%', maxWidth: '500px' }}>
              <h1 style={{ margin: '0 0 0.5rem', fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400', textAlign: 'center' }}>My <span style={{ color: accent, fontStyle: 'italic' }}>Profile</span></h1>
              <p style={{ color: subtext, marginBottom: '2rem', textAlign: 'center' }}>Your account details.</p>
              <div style={{ backgroundColor: card, borderRadius: '24px', border: `1px solid ${border}`, padding: '2rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '2rem' }}>
                  <div style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: accent, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.5rem', fontWeight: '700', color: '#000', flexShrink: 0 }}>{(userData?.name || currentUser?.email || 'P')[0].toUpperCase()}</div>
                  <div><div style={{ fontWeight: '600', fontSize: '1.1rem' }}>{userData?.name || 'Patient'}</div><div style={{ color: subtext, fontSize: '0.875rem' }}>{currentUser?.email}</div></div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {[{ label: 'Role', value: 'Patient' }, { label: 'Member Since', value: currentUser?.metadata?.creationTime ? new Date(currentUser.metadata.creationTime).getFullYear() : '2026' }, { label: 'Total Scans', value: String(scanHistory.length) }, { label: 'Appointments', value: String(appointments.length) }].map((item, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.8rem 0', borderBottom: `1px solid ${border}` }}>
                      <span style={{ color: subtext, fontSize: '0.9rem' }}>{item.label}</span>
                      <span style={{ fontWeight: '500', fontSize: '0.9rem' }}>{item.value}</span>
                    </div>
                  ))}
                </div>
                <button onClick={handleLogout} style={{ marginTop: '1.5rem', width: '100%', padding: '0.8rem', borderRadius: '999px', backgroundColor: 'transparent', color: '#ff4d4d', border: '1px solid #ff4d4d33', cursor: 'pointer', fontWeight: '600', fontFamily: 'inherit' }}>Sign Out</button>
              </div>
            </div>
          </div>
        )}

      </main>
      {/* ── DOCTOR SELECTION MODAL ──────────────────────── */}
      {showDoctorModal && (
        <div
          onClick={() => setShowDoctorModal(false)}
          style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '1rem' }}>
          <div
            onClick={e => e.stopPropagation()}
            style={{ backgroundColor: card, borderRadius: '28px', border: `1px solid ${border}`, padding: '2rem', width: '100%', maxWidth: '480px', maxHeight: '80vh', display: 'flex', flexDirection: 'column', boxShadow: '0 25px 60px rgba(0,0,0,0.5)' }}>

            {/* Modal header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
              <div>
                <h3 style={{ margin: '0 0 0.3rem', fontSize: '1.2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
                  Select a <span style={{ color: accent, fontStyle: 'italic' }}>Doctor</span>
                </h3>
                <p style={{ margin: 0, color: subtext, fontSize: '0.85rem' }}>{doctors.length} doctor{doctors.length !== 1 ? 's' : ''} available</p>
              </div>
              <button
                onClick={() => setShowDoctorModal(false)}
                style={{ background: 'none', border: `1px solid ${border}`, color: subtext, borderRadius: '50%', width: '32px', height: '32px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>

            {/* Doctor list — scrollable */}
            <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem', flex: 1 }}>
              {doctors.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: subtext }}>No doctors registered yet.</div>
              ) : doctors.map(d => {
                const isSelected = selectedDoctor?.id === d.id
                return (
                  <div key={d.id}
                    onClick={() => { setSelectedDoctor(d); setShowDoctorModal(false) }}
                    style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.2rem', borderRadius: '16px', border: `1.5px solid ${isSelected ? accent : border}`, backgroundColor: isSelected ? (darkMode ? '#1a2600' : '#f6ffe6') : bg, cursor: 'pointer', transition: 'all 0.2s' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
                      {/* Avatar */}
                      <div style={{ width: '42px', height: '42px', borderRadius: '50%', backgroundColor: isSelected ? accent : (darkMode ? '#222' : '#eee'), display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.1rem', flexShrink: 0, transition: 'all 0.2s' }}>
                        {isSelected
                          ? <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#000" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                          : <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={subtext} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>}
                      </div>
                      <div>
                        <div style={{ fontWeight: '600', fontSize: '0.95rem', color: text, marginBottom: '0.15rem' }}>{d.name}</div>
                        <div style={{ color: subtext, fontSize: '0.8rem' }}>{d.specialty || 'Cardiologist'}</div>
                        <div style={{ color: subtext, fontSize: '0.75rem' }}>{d.email}</div>
                      </div>
                    </div>
                    {isSelected && (
                      <span style={{ fontSize: '0.75rem', color: accent, fontWeight: '600', whiteSpace: 'nowrap', marginLeft: '0.5rem' }}>Selected</span>
                    )}
                  </div>
                )
              })}
            </div>

          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <ToastContainer toasts={toasts} dismiss={dismiss} darkMode={darkMode} />
    </div>
  )
}

export default PatientDashboard
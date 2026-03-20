import { useState, useEffect } from 'react'
import { useToast, ToastContainer } from './Toast'
import { useNavigate } from 'react-router-dom'
import { auth, db } from '../firebase'
import { signOut } from 'firebase/auth'
import { collection, addDoc, getDocs, query, orderBy, where, doc, updateDoc } from 'firebase/firestore'
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

function DoctorDashboard({ darkMode, setDarkMode }) {
  const navigate = useNavigate()
  const { currentUser, userData } = useAuth()

  const [activeTab, setActiveTab] = useState('New Scan')
  const { toasts, toast, dismiss } = useToast()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [scanHistory, setScanHistory] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(true)

  // ── Patients & appointments ───────────────────────────────
  const [appointments, setAppointments] = useState([])
  const [loadingApts, setLoadingApts] = useState(true)
  const [aptFilter, setAptFilter] = useState('pending')
  const [expandedApt, setExpandedApt] = useState(null) // appointment id with report expanded
  const [scheduling, setScheduling] = useState(null)  // appointment id being scheduled
  const [pickedTime, setPickedTime] = useState('')
  const [saving, setSaving] = useState(false)

  const accent  = darkMode ? '#d5ff5f' : '#bce236'
  const bg      = darkMode ? '#000000' : '#ffffff'
  const card    = darkMode ? '#111111' : '#f5f5f5'
  const card2   = darkMode ? '#1a1a1a' : '#eeeeee'
  const text    = darkMode ? '#ffffff' : '#111111'
  const subtext = darkMode ? '#9ca3af' : '#6b7280'
  const border  = darkMode ? '#222222' : '#e5e5e5'

  const navAnim     = useAnimate(0)
  const contentAnim = useAnimate(300)
  const resultAnim  = useAnimate(200)

  useEffect(() => {
    if (!currentUser) return
    fetchHistory()
    fetchAppointments()
  }, [currentUser])

  const fetchHistory = async () => {
    try {
      const q = query(
        collection(db, 'scans', currentUser.uid, 'history'),
        orderBy('timestamp', 'desc')
      )
      const snapshot = await getDocs(q)
      setScanHistory(snapshot.docs.map(d => d.data()))
    } catch (err) {
      console.error('Error fetching history:', err)
    } finally {
      setLoadingHistory(false)
    }
  }

  const fetchAppointments = async () => {
    try {
      const q = query(
        collection(db, 'appointments'),
        where('doctorId', '==', currentUser.uid),
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

  const rejectAppointment = async (aptId) => {
    try {
      await updateDoc(doc(db, 'appointments', aptId), { status: 'rejected' })
      setAppointments(prev => prev.map(a => a.id === aptId ? { ...a, status: 'rejected' } : a))
    } catch (err) {
      console.error('Error rejecting appointment:', err)
    }
  }

  const confirmAppointment = async (aptId) => {
    if (!pickedTime) return
    setSaving(true)
    try {
      await updateDoc(doc(db, 'appointments', aptId), {
        appointmentTime: new Date(pickedTime).toISOString(),
        status: 'scheduled',
      })
      setScheduling(null)
      setPickedTime('')
      await fetchAppointments()
    } catch (err) {
      toast('Error: ' + err.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  const markDone = async (aptId) => {
    await updateDoc(doc(db, 'appointments', aptId), { status: 'done' })
    await fetchAppointments()
  }

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
        risk_score: data.risk_score,
        label: data.label,
        confidence: data.confidence,
        conditions: data.conditions,
        recommendation: data.recommendation,
        gradcam: data.gradcam
      })

    } catch (err) {
      console.error('Failed to connect to backend:', err)
      toast('Failed to connect to the AI server. Please try again.', 'error')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleDownloadReport = () => {
    if (!results) return;

    const printWindow = window.open('', '_blank');
    const dateStr = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const doctorName = userData?.name || currentUser?.email || 'Dr. User';
    
    const printAccent = '#88b500'; 
    const riskColor = results.label === 'High Risk' ? '#ff4d4d' : printAccent;

    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <title>CardioRetina Clinical Report - ${doctorName}</title>
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
              <div class="subtitle">Clinical Cardiovascular Risk Screening Report</div>
            </div>
            <div style="text-align: right; color: #666; font-size: 14px;">
              Generated on<br/><strong>${dateStr}</strong>
            </div>
          </div>
          <div class="patient-info">
            <div class="info-block"><span class="info-label">Attending Doctor</span><span class="info-value">${doctorName}</span></div>
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
          <div class="footer">Disclaimer: This report was generated by an AI screening prototype (CardioRetina) to assist clinicians. It is intended for informational and educational purposes only and does not replace a formal medical diagnosis.</div>
        </body>
      </html>
    `;
    
    printWindow.document.write(html);
    printWindow.document.close();
    setTimeout(() => { printWindow.focus(); printWindow.print(); }, 500);
  }

  const pendingCount = appointments.filter(a => a.status === 'pending').length
  const filteredApts = appointments.filter(a => a.status === aptFilter)
  const tabs = ['New Scan', 'Patients', 'History', 'Profile']

  return (
    <div style={{
      minHeight: '100vh', backgroundColor: bg, color: text,
      fontFamily: 'Bricolage Grotesque, sans-serif',
      transition: 'background-color 0.3s ease'
    }}>

      <nav style={{
        ...navAnim,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '1.2rem 3rem', borderBottom: `1px solid ${border}`,
        position: 'sticky', top: 0, backgroundColor: bg, zIndex: 100,
      }}>
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
              fontSize: '0.9rem', fontFamily: 'inherit', cursor: 'pointer',
              transition: 'all 0.2s', position: 'relative'
            }}>
              {tab}
              {tab === 'Patients' && pendingCount > 0 && (
                <span style={{ position: 'absolute', top: '4px', right: '6px', width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#f59e0b' }} />
              )}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
          <div style={{ fontSize: '0.85rem', color: subtext }}>
            {userData?.name || currentUser?.email || 'Dr. User'}
          </div>
          <button
            onClick={() => setDarkMode(!darkMode)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.3rem', display: 'flex', alignItems: 'center', opacity: 0.7, transition: 'opacity 0.2s' }}
            onMouseEnter={e => e.currentTarget.style.opacity = 1}
            onMouseLeave={e => e.currentTarget.style.opacity = 0.7}
          >
            {darkMode ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={text} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            )}
          </button>
          <button onClick={handleLogout} style={{
            padding: '0.5rem 1.2rem', borderRadius: '999px',
            backgroundColor: 'transparent', color: '#ff4d4d',
            border: '1px solid #ff4d4d33', cursor: 'pointer',
            fontSize: '0.85rem', fontFamily: 'inherit', fontWeight: '500'
          }}>
            Sign Out
          </button>
        </div>
      </nav>

      <main style={{ maxWidth: '1100px', margin: '0 auto', padding: '3rem 2rem' }}>

        {/* ══════════════════════════════════════════════════
            TAB: NEW SCAN  (unchanged)
        ══════════════════════════════════════════════════ */}
        {activeTab === 'New Scan' && (
          <div style={contentAnim}>
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
                New <span style={{ color: accent, fontStyle: 'italic' }}>Scan</span>
              </h1>
              <p style={{ margin: '0.4rem 0 0', color: subtext, fontSize: '0.95rem' }}>
                Upload and analyze retinal fundus images for cardiovascular risk.
              </p>
            </div>

            {!preview && (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                style={{
                  border: `2px dashed ${dragOver ? accent : border}`,
                  borderRadius: '24px', padding: '4rem 2rem', textAlign: 'center',
                  backgroundColor: dragOver ? (darkMode ? '#111' : '#f9f9f9') : card,
                  transition: 'all 0.2s', cursor: 'pointer', position: 'relative'
                }}
              >
                <input type="file" accept="image/*" onChange={handleFileChange} style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%', height: '100%' }} />
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={dragOver ? accent : subtext} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '1rem' }}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <h3 style={{ margin: '0 0 0.4rem', fontSize: '1.1rem', fontWeight: '500', color: dragOver ? accent : text }}>
                  {dragOver ? 'Drop to upload' : 'Upload Fundus Image'}
                </h3>
                <p style={{ margin: 0, color: subtext, fontSize: '0.875rem' }}>
                  Drag and drop or click to browse — JPEG, PNG supported
                </p>
              </div>
            )}

            {preview && !results && (
              <div style={{ marginTop: '2rem', display: 'flex', gap: '2rem', alignItems: 'center', backgroundColor: card, padding: '2rem', borderRadius: '24px', border: `1px solid ${border}` }}>
                <img src={preview} alt="Preview" style={{ width: '160px', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '16px', border: `1px solid ${border}` }} />
                <div style={{ flex: 1 }}>
                  <h4 style={{ margin: '0 0 0.5rem', fontSize: '1.1rem', fontWeight: '600' }}>Image Ready</h4>
                  <p style={{ color: subtext, fontSize: '0.9rem', marginBottom: '1.5rem', lineHeight: 1.6 }}>
                    MobileNetV2 will process this image for microvascular abnormalities and cardiovascular biomarkers.
                  </p>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <button onClick={runAnalysis} disabled={isAnalyzing} style={{
                      padding: '0.8rem 2rem', borderRadius: '999px',
                      backgroundColor: isAnalyzing ? card2 : accent,
                      color: '#000', fontWeight: '600', border: 'none',
                      cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                      fontSize: '0.95rem', fontFamily: 'inherit',
                      display: 'flex', alignItems: 'center', gap: '0.5rem',
                      opacity: isAnalyzing ? 0.7 : 1, transition: 'all 0.2s'
                    }}>
                      {isAnalyzing ? (
                        <>
                          <span style={{ width: '16px', height: '16px', border: '2px solid #000', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.8s linear infinite' }}/>
                          Analyzing...
                        </>
                      ) : 'Run AI Analysis →'}
                    </button>
                    <button onClick={() => { setFile(null); setPreview(null) }} style={{ padding: '0.8rem 1.5rem', borderRadius: '999px', backgroundColor: 'transparent', color: subtext, border: `1px solid ${border}`, cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit' }}>
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            )}

            {results && !isAnalyzing && (
              <div style={{ ...resultAnim, marginTop: '2.5rem' }}>
                <h2 style={{ margin: '0 0 2rem', fontSize: '1.6rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
                  Clinical <span style={{ color: accent, fontStyle: 'italic' }}>Results</span>
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: '1.5rem' }}>
                  <div style={{ backgroundColor: card, padding: '2rem', borderRadius: '24px', border: `1px solid ${border}`, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1.5rem' }}>
                    <p style={{ margin: 0, fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Risk Score</p>
                    <RiskGauge score={results.risk_score} label={results.label} darkMode={darkMode} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div style={{ backgroundColor: card, padding: '1.5rem', borderRadius: '24px', border: `1px solid ${border}` }}>
                      <p style={{ margin: '0 0 1rem', fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Heatmap Analysis</p>
                      <div style={{ display: 'flex', gap: '1rem' }}>
                        <div style={{ flex: 1 }}>
                          <p style={{ fontSize: '0.75rem', color: subtext, margin: '0 0 0.5rem' }}>Original</p>
                          <img src={preview} style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '12px' }} alt="Original" />
                        </div>
                        <div style={{ flex: 1 }}>
                          <p style={{ fontSize: '0.75rem', color: accent, margin: '0 0 0.5rem' }}>Grad-CAM</p>
                          {results?.gradcam ? (
                            <img src={`data:image/jpeg;base64,${results.gradcam}`} style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '12px' }} alt="Grad-CAM" />
                          ) : (
                            <div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '12px', background: `linear-gradient(135deg, #ff4d4d22, #ff4d4d55)`, border: `1px solid #ff4d4d44`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                              <span style={{ fontSize: '0.75rem', color: '#ff4d4d' }}>No heatmap</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                    <div style={{ backgroundColor: card, padding: '1.5rem', borderRadius: '24px', border: `1px solid ${border}` }}>
                      <p style={{ margin: '0 0 0.8rem', fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Detected Biomarkers</p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.5rem' }}>
                        {results.conditions.map((c, i) => (
                          <span key={i} style={{ padding: '0.3rem 0.8rem', borderRadius: '999px', backgroundColor: '#ff4d4d22', color: '#ff4d4d', fontSize: '0.8rem', fontWeight: '500' }}>{c}</span>
                        ))}
                      </div>
                      <p style={{ margin: '0 0 0.8rem', fontSize: '0.8rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Recommendation</p>
                      <p style={{ margin: 0, fontSize: '0.9rem', color: text, lineHeight: 1.7, padding: '1rem', borderRadius: '12px', backgroundColor: bg, border: `1px solid ${border}` }}>
                        {results.recommendation}
                      </p>
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                  <button onClick={handleDownloadReport} style={{ padding: '0.8rem 2rem', borderRadius: '999px', backgroundColor: accent, color: '#000', fontWeight: '600', border: 'none', cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit' }}>
                    Download PDF Report →
                  </button>
                  <button onClick={() => { setFile(null); setPreview(null); setResults(null) }} style={{ padding: '0.8rem 2rem', borderRadius: '999px', backgroundColor: card, color: text, fontWeight: '500', border: `1px solid ${border}`, cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit' }}>
                    New Scan
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ══════════════════════════════════════════════════
            TAB: PATIENTS  (new)
        ══════════════════════════════════════════════════ */}
        {activeTab === 'Patients' && (
          <div style={contentAnim}>
            <div style={{ marginBottom: '2rem' }}>
              <h1 style={{ margin: 0, fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
                My <span style={{ color: accent, fontStyle: 'italic' }}>Patients</span>
              </h1>
              <p style={{ margin: '0.4rem 0 0', color: subtext, fontSize: '0.95rem' }}>
                Appointment requests assigned to you.
              </p>
            </div>

            {/* Stat cards — double as filter buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
              {[
                { label: 'Pending',   value: appointments.filter(a => a.status === 'pending').length,   color: '#f59e0b',  key: 'pending'   },
                { label: 'Scheduled', value: appointments.filter(a => a.status === 'scheduled').length, color: accent,     key: 'scheduled' },
                { label: 'Completed', value: appointments.filter(a => a.status === 'done').length,      color: subtext,    key: 'done'      },
                { label: 'Rejected',  value: appointments.filter(a => a.status === 'rejected').length,  color: '#ff4d4d',  key: 'rejected'  },
              ].map(st => (
                <div key={st.key} onClick={() => setAptFilter(st.key)}
                  style={{
                    backgroundColor: card, padding: '1.5rem', borderRadius: '20px', cursor: 'pointer', transition: 'all 0.2s',
                    border: `1.5px solid ${aptFilter === st.key ? st.color : border}`,
                  }}>
                  <p style={{ margin: '0 0 0.4rem', fontSize: '0.78rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{st.label}</p>
                  <p style={{ margin: 0, fontSize: '2rem', fontWeight: '700', color: st.color, fontFamily: 'Junicode, serif' }}>{st.value}</p>
                </div>
              ))}
            </div>

            {/* Patient cards */}
            {loadingApts ? (
              <div style={{ textAlign: 'center', padding: '3rem', color: subtext }}>Loading patients...</div>
            ) : filteredApts.length === 0 ? (
              <div style={{ backgroundColor: card, borderRadius: '24px', border: `1px solid ${border}`, padding: '3rem', textAlign: 'center', color: subtext }}>
                No {aptFilter} appointments.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {filteredApts.map(apt => {
                  const isHigh      = apt.scanResult?.label === 'High Risk'
                  const riskColor   = isHigh ? '#ff4d4d' : accent
                  const statusColor = apt.status === 'pending' ? '#f59e0b' : apt.status === 'scheduled' ? accent : subtext
                  return (
                    <div key={apt.id} style={{ backgroundColor: card, borderRadius: '24px', border: `1px solid ${border}`, padding: '1.8rem 2rem' }}>

                      {/* Header row */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.2rem' }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '700', fontSize: '1rem', marginBottom: '0.5rem' }}>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={subtext} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
                            </svg>
                            {apt.patientName}
                          </div>
                          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                            <span style={{ padding: '0.25rem 0.75rem', borderRadius: '999px', backgroundColor: isHigh ? '#ff4d4d22' : `${accent}22`, color: riskColor, fontSize: '0.78rem', fontWeight: '600' }}>
                              {apt.scanResult?.label} — {apt.scanResult?.riskScore}%
                            </span>
                            <span style={{ padding: '0.25rem 0.75rem', borderRadius: '999px', backgroundColor: `${statusColor}22`, color: statusColor, fontSize: '0.78rem', fontWeight: '600', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                              {apt.status === 'pending'   && <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>Pending</>}
                              {apt.status === 'scheduled' && <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>Scheduled</>}
                              {apt.status === 'done'      && <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>Completed</>}
                            </span>
                          </div>
                        </div>

                        {/* Confirmed time */}
                        {apt.appointmentTime && (
                          <div style={{ textAlign: 'right' }}>
                            <p style={{ margin: '0 0 0.2rem', fontSize: '0.75rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Appointment</p>
                            <p style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700', color: accent, fontFamily: 'Junicode, serif' }}>
                              {new Date(apt.appointmentTime).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
                            </p>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: subtext }}>
                              {new Date(apt.appointmentTime).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Conditions */}
                      {apt.scanResult?.conditions?.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '1.2rem' }}>
                          {apt.scanResult.conditions.map((c, i) => (
                            <span key={i} style={{ padding: '0.2rem 0.65rem', borderRadius: '999px', backgroundColor: isHigh ? '#ff4d4d11' : `${accent}11`, color: riskColor, fontSize: '0.75rem', border: `1px solid ${isHigh ? '#ff4d4d33' : `${accent}33`}` }}>{c}</span>
                          ))}
                        </div>
                      )}

                      {/* Requested date */}
                      <p style={{ margin: '0 0 1.2rem', fontSize: '0.82rem', color: subtext }}>
                        Requested {apt.createdAt?.toDate ? apt.createdAt.toDate().toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '—'}
                      </p>

                      {/* Actions + View Report */}
                      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
                        {apt.status === 'pending' && (
                          <>
                            <button onClick={() => { setScheduling(apt.id); setPickedTime('') }}
                              style={{ padding: '0.65rem 1.6rem', borderRadius: '999px', backgroundColor: accent, color: '#000', fontWeight: '600', border: 'none', cursor: 'pointer', fontSize: '0.9rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.45rem', transition: 'opacity 0.2s' }}>
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>
                              </svg>
                              Set Appointment Time
                            </button>
                            <button onClick={() => rejectAppointment(apt.id)}
                              style={{ padding: '0.65rem 1.6rem', borderRadius: '999px', backgroundColor: 'transparent', color: '#ff4d4d', fontWeight: '600', border: '1px solid #ff4d4d', cursor: 'pointer', fontSize: '0.9rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                              </svg>
                              Reject
                            </button>
                          </>
                        )}
                        {apt.status === 'scheduled' && (
                          <>
                            <button onClick={() => { setScheduling(apt.id); setPickedTime(apt.appointmentTime ? apt.appointmentTime.slice(0, 16) : '') }}
                              style={{ padding: '0.65rem 1.6rem', borderRadius: '999px', backgroundColor: 'transparent', color: accent, fontWeight: '600', border: `1px solid ${accent}`, cursor: 'pointer', fontSize: '0.9rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                              </svg>
                              Reschedule
                            </button>
                            <button onClick={() => markDone(apt.id)}
                              style={{ padding: '0.65rem 1.6rem', borderRadius: '999px', backgroundColor: 'transparent', color: subtext, fontWeight: '500', border: `1px solid ${border}`, cursor: 'pointer', fontSize: '0.9rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="20 6 9 17 4 12"/>
                              </svg>
                              Mark as Done
                            </button>
                          </>
                        )}
                        {apt.status === 'done' && (
                          <span style={{ fontSize: '0.85rem', color: subtext, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={subtext} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                            Appointment completed
                          </span>
                        )}
                        {apt.status === 'rejected' && (
                          <span style={{ fontSize: '0.85rem', color: '#ff4d4d', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff4d4d" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                            Appointment rejected
                          </span>
                        )}

                        {/* View Report toggle */}
                        <button onClick={() => {
                          const next = expandedApt === apt.id ? null : apt.id
                          setExpandedApt(next)
                          console.log('scanResult:', apt.scanResult)
                          console.log('imageUrl:', apt.scanResult?.imageUrl)
                          console.log('gradcam length:', apt.scanResult?.gradcam?.length)
                        }}
                          style={{ padding: '0.65rem 1.6rem', borderRadius: '999px', backgroundColor: expandedApt === apt.id ? `${accent}22` : 'transparent', color: expandedApt === apt.id ? accent : subtext, fontWeight: '500', border: `1px solid ${expandedApt === apt.id ? accent : border}`, cursor: 'pointer', fontSize: '0.9rem', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.45rem', marginLeft: 'auto', transition: 'all 0.2s' }}>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                          </svg>
                          {expandedApt === apt.id ? 'Hide Report' : 'View Report'}
                        </button>
                      </div>

                      {/* Expanded scan report */}
                      {expandedApt === apt.id && (
                        <div style={{ marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: `1px solid ${border}` }}>
                          <p style={{ margin: '0 0 1rem', fontSize: '0.78rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Scan Report</p>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1rem', marginBottom: '1rem' }}>
                            <div style={{ backgroundColor: bg, borderRadius: '16px', border: `1px solid ${border}`, padding: '1.2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                              <p style={{ margin: 0, fontSize: '0.72rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Risk Score</p>
                              <p style={{ margin: 0, fontSize: '2.2rem', fontWeight: '700', color: riskColor, fontFamily: 'Junicode, serif', lineHeight: 1 }}>
                                {typeof apt.scanResult?.riskScore === 'number' ? apt.scanResult.riskScore.toFixed(1) : apt.scanResult?.riskScore}%
                              </p>
                              <span style={{ padding: '0.2rem 0.7rem', borderRadius: '999px', backgroundColor: isHigh ? '#ff4d4d22' : `${accent}22`, color: riskColor, fontSize: '0.75rem', fontWeight: '600' }}>
                                {apt.scanResult?.label}
                              </span>
                            </div>
                            <div style={{ backgroundColor: bg, borderRadius: '16px', border: `1px solid ${border}`, padding: '1.2rem' }}>
                              <p style={{ margin: '0 0 0.75rem', fontSize: '0.72rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Detected Biomarkers</p>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                                {apt.scanResult?.conditions?.length > 0
                                  ? apt.scanResult.conditions.map((c, i) => (
                                      <span key={i} style={{ padding: '0.25rem 0.7rem', borderRadius: '999px', backgroundColor: isHigh ? '#ff4d4d22' : `${accent}22`, color: riskColor, fontSize: '0.78rem', border: `1px solid ${isHigh ? '#ff4d4d33' : `${accent}33`}` }}>{c}</span>
                                    ))
                                  : <span style={{ fontSize: '0.85rem', color: subtext }}>No biomarkers recorded</span>
                                }
                              </div>
                            </div>
                          </div>
                          {/* Retinal image + Grad-CAM heatmap */}
                          {(apt.scanResult?.imageb64 || apt.scanResult?.gradcam) && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                              {apt.scanResult?.imageb64 && (
                                <div style={{ backgroundColor: bg, borderRadius: '16px', border: `1px solid ${border}`, padding: '1.2rem' }}>
                                  <p style={{ margin: '0 0 0.75rem', fontSize: '0.72rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Original Retinal Scan</p>
                                  <img src={`data:image/jpeg;base64,${apt.scanResult.imageb64}`} alt="Retinal scan"
                                    style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '12px', border: `1px solid ${border}` }}
                                  />
                                </div>
                              )}
                              {apt.scanResult?.gradcam && (
                                <div style={{ backgroundColor: bg, borderRadius: '16px', border: `1px solid ${border}`, padding: '1.2rem' }}>
                                  <p style={{ margin: '0 0 0.75rem', fontSize: '0.72rem', color: accent, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Grad-CAM Heatmap</p>
                                  <img src={`data:image/jpeg;base64,${apt.scanResult.gradcam}`} alt="Grad-CAM" style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', borderRadius: '12px', border: `1px solid ${border}` }} />
                                </div>
                              )}
                            </div>
                          )}

                          {apt.scanResult?.recommendation && (
                            <div style={{ backgroundColor: bg, borderRadius: '16px', border: `1px solid ${border}`, padding: '1.2rem' }}>
                              <p style={{ margin: '0 0 0.5rem', fontSize: '0.72rem', color: subtext, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Clinical Recommendation</p>
                              <p style={{ margin: 0, fontSize: '0.9rem', color: text, lineHeight: 1.7 }}>{apt.scanResult.recommendation}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* ══════════════════════════════════════════════════
            TAB: HISTORY  (unchanged)
        ══════════════════════════════════════════════════ */}
        {activeTab === 'History' && (
          <div style={contentAnim}>
            <h1 style={{ margin: '0 0 0.5rem', fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
              Scan <span style={{ color: accent, fontStyle: 'italic' }}>History</span>
            </h1>
            <p style={{ color: subtext, marginBottom: '2rem' }}>Past retinal scans and clinical results.</p>
            {loadingHistory ? (
              <div style={{ textAlign: 'center', padding: '3rem', color: subtext }}>Loading history...</div>
            ) : scanHistory.length === 0 ? (
              <div style={{ backgroundColor: card, borderRadius: '24px', border: `1px solid ${border}`, padding: '3rem', textAlign: 'center', color: subtext }}>
                No scans yet. Run your first analysis to see history here.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {scanHistory.map((scan, i) => (
                  <div key={i} style={{ backgroundColor: card, padding: '1.5rem 2rem', borderRadius: '20px', border: `1px solid ${border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <p style={{ margin: '0 0 0.3rem', fontWeight: '600' }}>{scan.date}</p>
                      <p style={{ margin: 0, fontSize: '0.85rem', color: subtext }}>Cardiovascular Risk Scan</p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <span style={{ fontSize: '1.2rem', fontWeight: '700', color: scan.label === 'High Risk' ? '#ff4d4d' : accent, fontFamily: 'Junicode, serif' }}>
                        {scan.score}%
                      </span>
                      <span style={{ padding: '0.3rem 0.8rem', borderRadius: '999px', backgroundColor: scan.label === 'High Risk' ? '#ff4d4d22' : '#d5ff5f22', color: scan.label === 'High Risk' ? '#ff4d4d' : accent, fontSize: '0.8rem', fontWeight: '600' }}>
                        {scan.label}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ══════════════════════════════════════════════════
            TAB: PROFILE  (unchanged)
        ══════════════════════════════════════════════════ */}
        {activeTab === 'Profile' && (
          <div style={{ ...contentAnim, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: '100%', maxWidth: '500px' }}>
              <h1 style={{ margin: '0 0 0.5rem', fontSize: '2rem', fontFamily: 'Junicode, serif', fontWeight: '400', textAlign: 'center' }}>
                Doctor <span style={{ color: accent, fontStyle: 'italic' }}>Profile</span>
              </h1>
              <p style={{ color: subtext, marginBottom: '2rem', textAlign: 'center' }}>Your account and verification status.</p>
              <div style={{ backgroundColor: card, borderRadius: '24px', border: `1px solid ${border}`, padding: '2rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '2rem' }}>
                  <div style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: accent, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.5rem', fontWeight: '700', color: '#000', flexShrink: 0 }}>
                    {(userData?.name || currentUser?.email || 'D')[0].toUpperCase()}
                  </div>
                  <div>
                    <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>{userData?.name || 'Dr. User'}</div>
                    <div style={{ color: subtext, fontSize: '0.875rem' }}>{currentUser?.email}</div>
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {[
                    { label: 'Role',         value: 'Doctor' },
                    { label: 'Specialty',    value: userData?.specialty || '—' },
                    { label: 'Member Since', value: currentUser?.metadata?.creationTime ? new Date(currentUser.metadata.creationTime).getFullYear() : '2026' },
                    { label: 'Total Scans',  value: String(scanHistory.length) },
                    { label: 'Patients',     value: String(appointments.length) },
                    { label: 'Verification', value: userData?.verificationStatus === 'verified' ? 'Verified' : 'Pending' },
                  ].map((item, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.8rem 0', borderBottom: `1px solid ${border}` }}>
                      <span style={{ color: subtext, fontSize: '0.9rem' }}>{item.label}</span>
                      <span style={{ fontWeight: '500', fontSize: '0.9rem', color: item.label === 'Verification' ? (userData?.verificationStatus === 'verified' ? accent : '#f59e0b') : text }}>
                        {item.value}
                      </span>
                    </div>
                  ))}
                </div>
                <button onClick={handleLogout} style={{ marginTop: '1.5rem', width: '100%', padding: '0.8rem', borderRadius: '999px', backgroundColor: 'transparent', color: '#ff4d4d', border: '1px solid #ff4d4d33', cursor: 'pointer', fontWeight: '600', fontFamily: 'inherit' }}>
                  Sign Out
                </button>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* ── SCHEDULING MODAL ─────────────────────────────── */}
      {scheduling && (
        <div onClick={() => setScheduling(null)}
          style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '1rem' }}>
          <div onClick={e => e.stopPropagation()}
            style={{ backgroundColor: card, borderRadius: '28px', border: `1px solid ${border}`, padding: '2rem', width: '100%', maxWidth: '420px', boxShadow: '0 25px 60px rgba(0,0,0,0.5)' }}>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
              <div>
                <h3 style={{ margin: '0 0 0.3rem', fontSize: '1.2rem', fontFamily: 'Junicode, serif', fontWeight: '400' }}>
                  Set <span style={{ color: accent, fontStyle: 'italic' }}>Appointment</span>
                </h3>
                <p style={{ margin: 0, color: subtext, fontSize: '0.85rem' }}>
                  {appointments.find(a => a.id === scheduling)?.patientName}
                </p>
              </div>
              <button onClick={() => setScheduling(null)}
                style={{ background: 'none', border: `1px solid ${border}`, borderRadius: '50%', width: '32px', height: '32px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={subtext} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>

            <input
              type="datetime-local"
              value={pickedTime}
              min={new Date().toISOString().slice(0, 16)}
              onChange={e => setPickedTime(e.target.value)}
              style={{ width: '100%', backgroundColor: darkMode ? '#1a1a1a' : '#f5f5f5', border: `1px solid ${border}`, color: text, borderRadius: '12px', padding: '0.9rem 1rem', fontSize: '1rem', boxSizing: 'border-box', marginBottom: '1.2rem', fontFamily: 'inherit', outline: 'none' }}
            />

            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button onClick={() => confirmAppointment(scheduling)} disabled={!pickedTime || saving}
                style={{ flex: 1, padding: '0.8rem', borderRadius: '999px', backgroundColor: !pickedTime || saving ? card2 : accent, color: '#000', fontWeight: '600', border: 'none', cursor: !pickedTime || saving ? 'not-allowed' : 'pointer', fontSize: '0.95rem', fontFamily: 'inherit', opacity: saving ? 0.7 : 1, transition: 'all 0.2s' }}>
                {saving ? 'Saving...' : 'Confirm Appointment'}
              </button>
              <button onClick={() => setScheduling(null)}
                style={{ padding: '0.8rem 1.4rem', borderRadius: '999px', backgroundColor: 'transparent', color: subtext, border: `1px solid ${border}`, cursor: 'pointer', fontSize: '0.95rem', fontFamily: 'inherit' }}>
                Cancel
              </button>
            </div>

          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
      <ToastContainer toasts={toasts} dismiss={dismiss} darkMode={darkMode} />
    </div>
  )
}

export default DoctorDashboard
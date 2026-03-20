import { Routes, Route, Navigate } from 'react-router-dom'
import { useState } from 'react'
import { useAuth } from './context/AuthContext'

import Landing from './pages/Landing'
import Signup from './pages/Signup'
import Login from './pages/Login'
import DoctorDashboard from './pages/DoctorDashboard'
import PatientDashboard from './pages/PatientDashboard'

function ProtectedRoute({ children, requiredRole }) {
  const { currentUser, userData } = useAuth()

  if (!currentUser) return <Navigate to="/login" replace />
  if (userData && userData.role !== requiredRole) {
    return <Navigate to={`/dashboard/${userData.role}`} replace />
  }
  return children
}

function App() {
  const [darkMode, setDarkMode] = useState(true)

  return (
    <Routes>
      <Route path="/" element={<Landing darkMode={darkMode} setDarkMode={setDarkMode} />} />
      <Route path="/signup" element={<Signup darkMode={darkMode} setDarkMode={setDarkMode} />} />
      <Route path="/login" element={<Login darkMode={darkMode} setDarkMode={setDarkMode} />} />

      <Route path="/dashboard/doctor" element={
        <ProtectedRoute requiredRole="doctor">
          <DoctorDashboard darkMode={darkMode} setDarkMode={setDarkMode} />
        </ProtectedRoute>
      } />

      <Route path="/dashboard/patient" element={
        <ProtectedRoute requiredRole="patient">
          <PatientDashboard darkMode={darkMode} setDarkMode={setDarkMode} />
        </ProtectedRoute>
      } />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
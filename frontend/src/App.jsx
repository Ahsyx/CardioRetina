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
  const [darkMode, setDarkMode]     = useState(true)
  const [purpleMode, setPurpleMode] = useState(false)

  const themeProps = { darkMode, setDarkMode, purpleMode, setPurpleMode }

  return (
    <Routes>
      <Route path="/"       element={<Landing          {...themeProps} />} />
      <Route path="/signup" element={<Signup           {...themeProps} />} />
      <Route path="/login"  element={<Login            {...themeProps} />} />

      <Route path="/dashboard/doctor" element={
        <ProtectedRoute requiredRole="doctor">
          <DoctorDashboard {...themeProps} />
        </ProtectedRoute>
      } />

      <Route path="/dashboard/patient" element={
        <ProtectedRoute requiredRole="patient">
          <PatientDashboard {...themeProps} />
        </ProtectedRoute>
      } />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
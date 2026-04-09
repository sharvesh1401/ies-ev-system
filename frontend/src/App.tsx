import { Suspense, lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import { VehicleProvider } from './contexts/VehicleContext'

// Lazy loading pages for performance optimization
const Home = lazy(() => import('./pages/Home'))
const RoutePlanner = lazy(() => import('./pages/RoutePlanner'))
const BatteryAnalytics = lazy(() => import('./pages/BatteryAnalytics'))
const ChargingStations = lazy(() => import('./pages/ChargingStations'))
const SystemInsights = lazy(() => import('./pages/SystemInsights'))
const Profile = lazy(() => import('./pages/Profile'))
const Settings = lazy(() => import('./pages/Settings'))

function App() {
  return (
    <VehicleProvider>
      <Suspense fallback={<div className="h-screen w-screen bg-brand-bg flex items-center justify-center"><div className="w-8 h-8 rounded-full border-t-2 border-r-2 border-neon-cyan animate-spin" /></div>}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Home />} />
            <Route path="/route-planner" element={<RoutePlanner />} />
            <Route path="/battery-analytics" element={<BatteryAnalytics />} />
            <Route path="/charging-stations" element={<ChargingStations />} />
            <Route path="/system-insights" element={<SystemInsights />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </VehicleProvider>
  )
}

export default App


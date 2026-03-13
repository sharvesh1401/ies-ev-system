import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import RoutePlanner from './pages/RoutePlanner'
import BatteryAnalytics from './pages/BatteryAnalytics'
import ChargingStations from './pages/ChargingStations'
import SystemInsights from './pages/SystemInsights'

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/route-planner" element={<RoutePlanner />} />
        <Route path="/battery-analytics" element={<BatteryAnalytics />} />
        <Route path="/charging-stations" element={<ChargingStations />} />
        <Route path="/system-insights" element={<SystemInsights />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App


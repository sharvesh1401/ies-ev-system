import { useState, useEffect } from 'react'
import { useVehicle } from '../contexts/VehicleContext'

export default function TopBar() {
  const [time, setTime] = useState(new Date())
  const { vehicle } = useVehicle()
  const soc = vehicle.battery.soc_percent

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const hours = time.getHours().toString().padStart(2, '0')
  const minutes = time.getMinutes().toString().padStart(2, '0')
  const seconds = time.getSeconds().toString().padStart(2, '0')

  const batteryIcon = soc > 60 ? 'battery_full' : soc > 30 ? 'battery_4_bar' : 'battery_1_bar'

  return (
    <header className="h-14 flex items-center justify-between px-8 z-40 shrink-0 border-b border-white/5 bg-surface-container-highest/60 backdrop-blur-xl shadow-2xl shadow-black/40">
      {/* Left: System status */}
      <div className="flex items-center gap-2 font-mono text-xs font-bold uppercase tracking-widest text-green-400">
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
        System Status: CONNECTED
      </div>

      {/* Right: icons + clock */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4 text-on-surface-variant">
          <span className="material-symbols-outlined text-sm hover:text-primary transition-colors cursor-pointer">wifi</span>
          <span className="material-symbols-outlined text-sm hover:text-primary transition-colors cursor-pointer">{batteryIcon}</span>
        </div>
        <div className="h-4 w-px bg-white/10" />
        <div className="font-mono text-xs font-bold tracking-widest text-primary">
          {hours}:{minutes}:{seconds}
        </div>
      </div>
    </header>
  )
}

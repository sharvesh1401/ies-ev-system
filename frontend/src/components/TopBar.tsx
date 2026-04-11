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
    <header className="h-16 flex items-center justify-between px-8 z-40 shrink-0 border-b border-outline-variant bg-surface-container-low/80 backdrop-blur-xl">
      {/* Left: System status */}
      <div className="flex items-center gap-2 font-mono text-xs font-bold uppercase tracking-widest text-accent-success">
        <span className="w-2 h-2 rounded-full bg-accent-success animate-pulse" />
        System Status: CONNECTED
      </div>

      {/* Right: icons + clock */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4 text-on-surface">
           <span className="material-symbols-outlined text-base hover:text-primary transition-colors duration-150 cursor-pointer">wifi</span>
           <span className="material-symbols-outlined text-base hover:text-primary transition-colors duration-150 cursor-pointer">{batteryIcon}</span>
        </div>
        <div className="h-5 w-px bg-outline-variant" />
        <div className="font-mono text-xs font-bold tracking-widest text-primary tabular-nums">
          {hours}:{minutes}:{seconds}
        </div>
      </div>
    </header>
  )
}

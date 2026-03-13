import { useState, useEffect } from 'react'

export default function TopBar() {
  const [time, setTime] = useState(new Date())
  const soc = 76 // simulated

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const hours = time.getHours().toString().padStart(2, '0')
  const minutes = time.getMinutes().toString().padStart(2, '0')
  const seconds = time.getSeconds().toString().padStart(2, '0')

  // SoC color coding per spec
  const socColor = soc > 60 ? 'text-accent-success' : soc > 30 ? 'text-neon-yellow' : 'text-accent-danger'
  const batteryIcon = soc > 60 ? 'battery_charging_full' : soc > 30 ? 'battery_4_bar' : 'battery_1_bar'

  return (
    <header className="h-14 flex items-center justify-between px-8 z-10 shrink-0 border-b border-neon-cyan/8 bg-surface-100/50 backdrop-blur-md">
      <div className="flex items-center gap-4">
        <h2 className="text-sm font-headline font-semibold text-surface-900">
          System Status
        </h2>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-accent-success/10 border border-accent-success/15">
          <div className="w-1.5 h-1.5 rounded-full bg-accent-success animate-pulse" />
          <span className="text-[9px] font-bold text-accent-success uppercase tracking-widest">Connected</span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        {/* Live Clock */}
        <div className="flex items-center gap-2 font-mono text-sm">
          <span className="text-neon-cyan font-bold tabular-nums">{hours}</span>
          <span className="text-neon-cyan/40 animate-pulse">:</span>
          <span className="text-neon-cyan font-bold tabular-nums">{minutes}</span>
          <span className="text-neon-cyan/40 animate-pulse">:</span>
          <span className="text-neon-cyan/50 font-bold tabular-nums text-xs">{seconds}</span>
        </div>

        <div className="w-px h-5 bg-surface-200/30" />

        <div className="flex items-center gap-3 text-surface-800/50">
          <span className="material-symbols-outlined text-base">wifi</span>
          <div className="flex items-center gap-1.5">
            <span
              className={`material-symbols-outlined ${socColor} text-base`}
              style={{ fontVariationSettings: "'FILL' 1" }}
            >
              {batteryIcon}
            </span>
            <span className={`text-xs font-bold tabular-nums ${socColor}`}>{soc}%</span>
          </div>
        </div>
      </div>
    </header>
  )
}

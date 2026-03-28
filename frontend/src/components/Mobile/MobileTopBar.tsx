import { useState, useEffect } from 'react'

export default function MobileTopBar() {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const hours = time.getHours().toString().padStart(2, '0')
  const minutes = time.getMinutes().toString().padStart(2, '0')

  return (
    <header className="fixed top-0 left-0 right-0 h-16 z-[100] flex items-center justify-between px-4 bg-surface-100/90 backdrop-blur-xl border-b border-neon-cyan/10 md:hidden"
      style={{ paddingTop: 'env(safe-area-inset-top)' }}
    >
      {/* Left: Logo + Title */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-[10px] overflow-hidden shrink-0 glow-cyan">
          <img src="/logo.png" alt="Meridian Logo" className="w-full h-full object-cover" />
        </div>
        <span className="text-lg font-headline font-bold text-surface-900 tracking-tight">Meridian</span>
      </div>

      {/* Right: Status Icons */}
      <div className="flex items-center gap-3">
        <span className="material-symbols-outlined text-surface-800/50 text-lg">wifi</span>

        <div className="flex items-center gap-1.5 font-mono text-sm">
          <span className="text-neon-cyan font-bold tabular-nums">{hours}</span>
          <span className="text-neon-cyan/40 animate-pulse">:</span>
          <span className="text-neon-cyan font-bold tabular-nums">{minutes}</span>
        </div>

        <div className="flex items-center gap-1">
          <span
            className="material-symbols-outlined text-accent-success text-lg"
            style={{ fontVariationSettings: "'FILL' 1" }}
          >
            battery_charging_full
          </span>
          <span className="text-xs font-bold tabular-nums text-accent-success">98%</span>
        </div>
      </div>
    </header>
  )
}

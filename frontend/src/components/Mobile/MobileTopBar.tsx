import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

export default function MobileTopBar() {
  const [time, setTime]           = useState(new Date())
  const [profileOpen, setProfile] = useState(false)
  const [isLight, setIsLight]     = useState(false)
  const navigate = useNavigate()

  // Live clock
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  // Sync initial theme state with whatever the sidebar may have set
  useEffect(() => {
    setIsLight(document.documentElement.classList.contains('light'))
  }, [])

  const hours   = time.getHours().toString().padStart(2, '0')
  const minutes = time.getMinutes().toString().padStart(2, '0')

  function toggleTheme() {
    const root = document.documentElement
    if (root.classList.contains('light')) {
      root.classList.remove('light')
      setIsLight(false)
    } else {
      root.classList.add('light')
      setIsLight(true)
    }
  }

  function go(path: string) {
    setProfile(false)
    navigate(path)
  }

  return (
    <>
      {/* ── Fixed Top Bar ── */}
      <header
        className="fixed top-0 left-0 right-0 z-[100] flex items-center justify-between px-4 bg-surface-100/90 backdrop-blur-xl border-b border-neon-cyan/10 md:hidden"
        style={{
          height: 'calc(4rem + env(safe-area-inset-top))',
          paddingTop: 'env(safe-area-inset-top)',
        }}
      >
        {/* Left: Logo + Title */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-[10px] overflow-hidden shrink-0 glow-cyan">
            <img src="/logo.png" alt="Meridian Logo" className="w-full h-full object-cover" />
          </div>
          <span className="text-lg font-headline font-bold text-surface-900 tracking-tight">Meridian</span>
        </div>

        {/* Right: time + battery + profile avatar */}
        <div className="flex items-center gap-3">
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

          {/* Profile avatar — opens slide-up sheet */}
          <button
            onClick={() => setProfile(true)}
            aria-label="Open profile menu"
            className="w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-bold text-white shrink-0 active:scale-90 transition-transform"
            style={{
              background: 'linear-gradient(135deg, #A855F7, #7C3AED)',
              boxShadow: '0 0 10px rgba(168,85,247,0.4)',
            }}
          >
            SH
          </button>
        </div>
      </header>

      {/* ── Profile Sheet ── */}
      {profileOpen && (
        <>
          {/* Scrim */}
          <div
            className="fixed inset-0 z-[200] bg-black/50 md:hidden"
            onClick={() => setProfile(false)}
          />

          {/* Sheet */}
          <div
            className="fixed left-0 right-0 bottom-0 z-[201] bg-surface-container-high rounded-t-3xl md:hidden overflow-hidden"
            style={{ animation: 'sheetUp 220ms cubic-bezier(0.32,0.72,0,1)' }}
          >
            {/* Drag handle */}
            <div className="flex justify-center pt-3 pb-1">
              <div className="w-10 h-1 rounded-full bg-on-surface-variant/30" />
            </div>

            {/* User header → tapping goes to /profile */}
            <div
              className="mx-3 mt-1 mb-2 p-4 rounded-2xl border border-outline-variant/30 cursor-pointer active:bg-surface-variant/40 transition-colors"
              onClick={() => go('/profile')}
            >
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-full bg-primary flex items-center justify-center text-sm font-bold text-stitch-on-primary shrink-0">
                  SH
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-bold text-on-surface text-sm">Sharvesh</p>
                  <p className="text-[10px] font-mono uppercase tracking-[0.15em] text-on-surface-variant font-bold">System Developer</p>
                  <p className="text-[11px] text-on-surface-variant/80 truncate">s_sharvesh@outlook.com</p>
                </div>
                <span className="material-symbols-outlined text-on-surface-variant text-xl">chevron_right</span>
              </div>
            </div>

            {/* Menu items */}
            <div className="px-3 pb-2 space-y-0.5">
              {/* Settings */}
              <button
                onClick={() => go('/settings')}
                className="w-full flex items-center justify-between px-4 py-3.5 rounded-xl text-left active:bg-surface-variant/40 transition-colors group"
              >
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-on-surface-variant text-[22px]">settings</span>
                  <span className="text-sm font-medium text-on-surface">Settings</span>
                </div>
                <span className="material-symbols-outlined text-on-surface-variant">chevron_right</span>
              </button>

              {/* Notifications */}
              <button className="w-full flex items-center justify-between px-4 py-3.5 rounded-xl text-left active:bg-surface-variant/40 transition-colors">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-on-surface-variant text-[22px]">notifications</span>
                  <span className="text-sm font-medium text-on-surface">Notifications</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="bg-primary text-stitch-on-primary text-[10px] font-mono font-bold px-2 py-0.5 rounded-full min-w-[22px] text-center">3</span>
                  <span className="material-symbols-outlined text-on-surface-variant">chevron_right</span>
                </div>
              </button>

              {/* Theme toggle */}
              <div className="flex items-center justify-between px-4 py-3.5 rounded-xl">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-on-surface-variant text-[22px]">
                    {isLight ? 'light_mode' : 'dark_mode'}
                  </span>
                  <span className="text-sm font-medium text-on-surface">Theme Mode</span>
                </div>
                <button
                  onClick={toggleTheme}
                  className={`relative w-[52px] h-[28px] rounded-full p-1 transition-colors duration-300 ${
                    isLight ? 'bg-primary/20' : 'bg-surface-container-highest shadow-inner'
                  }`}
                >
                  <div
                    className={`absolute top-1 left-1 w-5 h-5 rounded-full flex items-center justify-center transition-all duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
                      isLight
                        ? 'translate-x-[24px] bg-primary text-on-primary shadow-[0_2px_8px_rgba(0,217,255,0.4)]'
                        : 'translate-x-0 bg-on-surface-variant text-surface-container-highest shadow-[0_2px_5px_rgba(0,0,0,0.5)]'
                    }`}
                  >
                    <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'FILL' 1" }}>
                      {isLight ? 'light_mode' : 'dark_mode'}
                    </span>
                  </div>
                </button>
              </div>
            </div>

            {/* Sign out */}
            <div className="px-3 py-2 border-t border-outline-variant/30">
              <button className="w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-left active:bg-error/10 transition-colors">
                <span className="material-symbols-outlined text-error text-[22px]">logout</span>
                <span className="text-sm text-error font-medium">Sign Out</span>
              </button>
            </div>

            {/* Footer */}
            <div
              className="px-5 py-3 border-t border-outline-variant/30 flex justify-between text-[9px] font-mono text-on-surface-variant/70 uppercase tracking-widest bg-surface-container-lowest"
              style={{ paddingBottom: 'calc(0.75rem + env(safe-area-inset-bottom))' }}
            >
              <span>Last Login: {new Date().toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' })}</span>
              <span>v.4.2.0-stable</span>
            </div>
          </div>
        </>
      )}

      {/* Sheet slide-up keyframe (only needed on mobile) */}
      <style>{`
        @keyframes sheetUp {
          from { transform: translateY(100%); }
          to   { transform: translateY(0); }
        }
      `}</style>
    </>
  )
}

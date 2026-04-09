import { useState, useRef, useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'

const navItems = [
  { to: '/', icon: 'home', label: 'Home' },
  { to: '/route-planner', icon: 'route', label: 'Route Planner' },
  { to: '/battery-analytics', icon: 'battery_charging_full', label: 'Battery Analytics' },
  { to: '/charging-stations', icon: 'ev_station', label: 'Charging Stations' },
  { to: '/system-insights', icon: 'monitoring', label: 'System Insights' },
]

export default function Sidebar() {
  const navigate = useNavigate()
  const [profileOpen, setProfileOpen] = useState(false)
  const [isLight, setIsLight] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLDivElement>(null)

  const toggleTheme = () => {
    const root = document.documentElement
    if (root.classList.contains('light')) {
      root.classList.remove('light')
      setIsLight(false)
    } else {
      root.classList.add('light')
      setIsLight(true)
    }
  }

  useEffect(() => {
    if (!profileOpen) return
    function handleClick(e: MouseEvent) {
      const t = e.target as Node
      if (!dropdownRef.current?.contains(t) && !triggerRef.current?.contains(t)) {
        setProfileOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [profileOpen])

  return (
    <aside className="w-[260px] hidden md:flex flex-col h-full z-50 shrink-0 bg-[#181c24] border-r border-white/5">

      {/* ── Logo ── */}
      <div className="px-4 pt-6">
        <div className="bg-surface-container border border-outline-variant/20 rounded-2xl px-5 py-4">
          <h1 className="text-[32px] font-bold tracking-tight text-primary leading-none">
            Meridian
          </h1>
          <p className="text-[12px] font-medium text-on-surface-variant/70 mt-1.5 leading-snug">
            Automotive Intelligence
          </p>
        </div>
      </div>

      {/* Spacer between logo and nav */}
      <div className="h-20 shrink-0" />

      {/* ── Navigation ── */}
      <nav className="flex flex-col gap-[36px] pl-8 pr-0">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `w-full flex items-center gap-5 transition-colors duration-200 relative ${
                isActive
                  ? 'text-primary font-semibold'
                  : 'text-on-surface-variant hover:text-on-surface font-medium'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className="material-symbols-outlined text-[48px] shrink-0"
                  style={{ fontVariationSettings: isActive ? "'FILL' 1, 'wght' 500" : "'FILL' 0, 'wght' 300" }}
                >
                  {item.icon}
                </span>
                <span className="text-[22px] font-medium">{item.label}</span>
                
                {/* Right-edge active indicator */}
                {isActive && (
                  <span className="absolute right-0 top-1/2 -translate-y-1/2 h-8 w-[4px] bg-primary rounded-l-md" />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Spacer to push user block to bottom */}
      <div className="flex-1" />

      {/* ── User Block ── */}
      <div className="px-3 pt-4 pb-5 border-t border-[var(--stitch-outline-variant)] relative">
        <div
          ref={triggerRef}
          onClick={() => setProfileOpen(!profileOpen)}
          className="flex items-center gap-3 px-3 py-3 rounded-xl cursor-pointer hover:bg-surface-variant/40 transition-all duration-200"
        >
          {/* Avatar */}
          <div className="w-11 h-11 rounded-full bg-surface-container border border-[var(--stitch-outline-variant)] flex items-center justify-center shrink-0">
            <span
              className="material-symbols-outlined text-on-surface-variant text-[28px]"
              style={{ fontVariationSettings: "'FILL' 1" }}
            >
              person
            </span>
          </div>

          {/* Name + role */}
          <div className="flex-1 min-w-0">
            <p className="text-[14px] font-semibold text-on-surface leading-tight">Sharvesh</p>
            <p className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant mt-0.5 font-bold">
              Lead Architect
            </p>
          </div>
        </div>

        {/* ── Profile Dropdown ── */}
        {profileOpen && (
          <div
            ref={dropdownRef}
            className="absolute bottom-[84px] left-3 right-3 bg-surface-container-high border border-[var(--stitch-outline-variant)] rounded-2xl shadow-xl overflow-hidden z-[100]"
            style={{ animation: 'dropdownIn 150ms ease-out', transformOrigin: 'bottom center' }}
          >
            {/* User header */}
            <div
              className="p-4 border-b border-[var(--stitch-outline-variant)] cursor-pointer hover:bg-surface-variant/40 transition-colors duration-150"
              onClick={() => { navigate('/profile'); setProfileOpen(false) }}
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-sm font-bold text-stitch-on-primary shrink-0">
                  SH
                </div>
                <div>
                  <p className="font-bold text-on-surface text-sm">Sharvesh</p>
                  <p className="text-[10px] font-mono uppercase tracking-[0.15em] text-on-surface-variant font-bold">System Developer</p>
                  <p className="text-[11px] text-on-surface-variant/80">s_sharvesh@outlook.com</p>
                </div>
              </div>
            </div>

            {/* Menu items */}
            <div className="p-2">
              <button
                onClick={() => { navigate('/profile'); setProfileOpen(false) }}
                className="w-full flex items-center justify-between px-3 py-2.5 rounded-[10px] text-left hover:bg-surface-variant/40 transition-colors duration-150 group"
              >
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-on-surface-variant group-hover:text-primary transition-colors duration-150">settings</span>
                  <span className="text-sm font-medium text-on-surface">Settings</span>
                </div>
                <span className="material-symbols-outlined text-on-surface-variant">chevron_right</span>
              </button>

              <button className="w-full flex items-center justify-between px-3 py-2.5 rounded-[10px] text-left hover:bg-surface-variant/40 transition-colors duration-150 group">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-on-surface-variant group-hover:text-primary transition-colors duration-150">notifications</span>
                  <span className="text-sm font-medium text-on-surface">Notifications</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="bg-primary text-stitch-on-primary text-[10px] font-mono font-bold px-2 py-0.5 rounded-full min-w-[22px] text-center">3</span>
                  <span className="material-symbols-outlined text-on-surface-variant">chevron_right</span>
                </div>
              </button>

              {/* Theme Toggle */}
              <div className="flex items-center justify-between px-3 py-2.5 rounded-[10px] mt-1">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-on-surface-variant">dark_mode</span>
                  <span className="text-sm font-medium text-on-surface">Theme Mode</span>
                </div>
                <div className="flex bg-surface-container-lowest rounded-[10px] overflow-hidden border border-[var(--stitch-outline-variant)] p-0.5 gap-0.5">
                  <button
                    onClick={() => { if (isLight) toggleTheme() }}
                    className={`px-3 py-1 text-[10px] rounded-lg font-mono font-bold uppercase tracking-widest transition-all duration-150 ${
                      !isLight ? 'bg-primary text-stitch-on-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    Dark
                  </button>
                  <button
                    onClick={() => { if (!isLight) toggleTheme() }}
                    className={`px-3 py-1 text-[10px] rounded-lg font-mono font-bold uppercase tracking-widest transition-all duration-150 ${
                      isLight ? 'bg-primary text-stitch-on-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    Light
                  </button>
                </div>
              </div>
            </div>

            {/* Sign Out */}
            <div className="p-2 border-t border-[var(--stitch-outline-variant)]">
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-[10px] text-left hover:bg-error/10 transition-colors duration-150 group">
                <span className="material-symbols-outlined text-error group-hover:text-error/80">logout</span>
                <span className="text-sm text-error font-medium group-hover:text-error/80">Sign Out</span>
              </button>
            </div>

            {/* Footer */}
            <div className="px-4 py-3 border-t border-[var(--stitch-outline-variant)] flex justify-between text-[9px] font-mono text-on-surface-variant/70 uppercase tracking-widest bg-surface-container-lowest">
              <span>Last Login: {new Date().toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' })}</span>
              <span>v.4.2.0-stable</span>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

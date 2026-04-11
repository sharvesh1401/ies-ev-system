import { useState, useRef, useEffect } from 'react'
import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import { useVehicle } from '../contexts/VehicleContext'

const navItems = [
  { to: '/', icon: 'home', label: 'Home' },
  { to: '/route-planner', icon: 'route', label: 'Route Planner' },
  { to: '/battery-analytics', icon: 'battery_charging_full', label: 'Battery Analytics' },
  { to: '/charging-stations', icon: 'ev_station', label: 'Charging Stations' },
  { to: '/system-insights', icon: 'monitoring', label: 'System Insights' },
]

export default function Sidebar() {
  const navigate = useNavigate()
  const location = useLocation()
  const { currentVehicle, switchVehicle, setLabMinimized } = useVehicle()
  const [profileOpen, setProfileOpen] = useState(false)
  const [isLight, setIsLight] = useState(() => {
    return localStorage.getItem('theme') === 'light' || document.documentElement.classList.contains('light')
  })
  const dropdownRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isLight) {
      document.documentElement.classList.add('light')
    } else {
      document.documentElement.classList.remove('light')
    }
  }, [isLight])

  const toggleTheme = () => {
    const root = document.documentElement
    if (root.classList.contains('light')) {
      root.classList.remove('light')
      localStorage.setItem('theme', 'dark')
      setIsLight(false)
    } else {
      root.classList.add('light')
      localStorage.setItem('theme', 'light')
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
    <aside className={`w-[260px] hidden md:flex flex-col h-full z-50 shrink-0 border-r ${isLight ? 'bg-white border-outline-variant/10' : 'bg-[#181c24] border-white/5'}`}>

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
        {/* Home */}
        <button
          onClick={() => {
            if (currentVehicle === 'custom-lab') {
              switchVehicle('model-v-performance')
              setLabMinimized(false)
            }
            navigate('/')
          }}
          className={`w-full flex items-center gap-5 transition-colors duration-200 relative ${
            location.pathname === '/' && currentVehicle !== 'custom-lab'
              ? 'text-primary font-semibold'
              : 'text-on-surface-variant hover:text-on-surface font-medium'
          }`}
        >
          <span
            className="material-symbols-outlined text-[48px] shrink-0"
            style={{ fontVariationSettings: location.pathname === '/' && currentVehicle !== 'custom-lab' ? "'FILL' 1, 'wght' 500" : "'FILL' 0, 'wght' 300" }}
          >
            home
          </span>
          <span className="text-[22px] font-medium">Home</span>
          {location.pathname === '/' && currentVehicle !== 'custom-lab' && (
            <span className="absolute right-0 top-1/2 -translate-y-1/2 h-8 w-[4px] bg-primary rounded-l-md" />
          )}
        </button>

        {/* Custom Lab */}
        <button
          onClick={() => {
            switchVehicle('custom-lab')
            setLabMinimized(false)
            navigate('/')
          }}
          className={`w-full flex items-center gap-5 transition-colors duration-200 relative ${
            currentVehicle === 'custom-lab'
              ? 'font-semibold'
              : 'text-on-surface-variant hover:text-on-surface font-medium'
          }`}
          style={{ color: currentVehicle === 'custom-lab' ? '#A855F7' : undefined }}
        >
          <span
            className="material-symbols-outlined text-[48px] shrink-0"
            style={{ fontVariationSettings: currentVehicle === 'custom-lab' ? "'FILL' 1, 'wght' 500" : "'FILL' 0, 'wght' 300" }}
          >
            science
          </span>
          <span className="text-[22px] font-medium">Custom Lab</span>
          {currentVehicle === 'custom-lab' && (
            <span className="absolute right-0 top-1/2 -translate-y-1/2 h-8 w-[4px] rounded-l-md" style={{ background: '#A855F7' }} />
          )}
        </button>

        {/* Remaining nav items */}
        {navItems.slice(1).map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
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
                onClick={() => { navigate('/settings'); setProfileOpen(false) }}
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

              {/* Premium Theme Toggle */}
              <div className="flex items-center justify-between px-3 py-2.5 rounded-[10px] mt-1 group">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-on-surface-variant group-hover:text-primary transition-colors duration-200">
                    {isLight ? 'light_mode' : 'dark_mode'}
                  </span>
                  <span className="text-sm font-medium text-on-surface">Theme Mode</span>
                </div>
                
                {/* Premium Slider Track */}
                <button
                  onClick={toggleTheme}
                  className={`relative w-[52px] h-[28px] rounded-full p-1 transition-colors duration-300 ${
                    isLight ? 'bg-primary/20 shadow-[inset_0_2px_4px_rgba(0,217,255,0.1)]' : 'bg-surface-container-highest shadow-inner'
                  }`}
                >
                  {/* Slider Thumb */}
                  <div
                    className={`absolute top-1 left-1 w-5 h-5 rounded-full flex items-center justify-center transition-all duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
                      isLight 
                        ? 'translate-x-[24px] bg-primary text-on-primary shadow-[0_2px_8px_rgba(0,217,255,0.4)]' 
                        : 'translate-x-0 bg-on-surface-variant text-surface-container-highest shadow-[0_2px_5px_rgba(0,0,0,0.5)]'
                    }`}
                  >
                    <span 
                      className="material-symbols-outlined text-[14px]"
                      style={{ fontVariationSettings: "'FILL' 1" }}
                    >
                      {isLight ? 'light_mode' : 'dark_mode'}
                    </span>
                  </div>
                </button>
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

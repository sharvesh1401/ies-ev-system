import { useState } from 'react'
import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', icon: 'home', label: 'Home' },
  { to: '/route-planner', icon: 'route', label: 'Route Planner' },
  { to: '/battery-analytics', icon: 'battery_charging_full', label: 'Battery Analytics' },
  { to: '/charging-stations', icon: 'ev_station', label: 'Charging Stations' },
  { to: '/system-insights', icon: 'monitoring', label: 'System Insights' },
]

export default function Sidebar() {
  const [profileOpen, setProfileOpen] = useState(false)
  const [isLight, setIsLight] = useState(false)

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

  return (
    <aside className="w-[260px] hidden md:flex flex-col h-full z-50 shrink-0 bg-surface-container-low border-r border-white/5">
      {/* Logo */}
      <div className="px-6 py-6 mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg overflow-hidden shrink-0">
            <img src="/logo.png" alt="Meridian Logo" className="w-full h-full object-cover" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tighter text-primary leading-none">
              Meridian
            </h1>
            <p className="text-sm font-medium tracking-tight text-on-surface/60 mt-0.5">
              Automotive Intelligence
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-0">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center px-6 py-3 space-x-3 text-sm font-medium tracking-tight transition-all duration-200 ease-in-out ${
                isActive
                  ? 'text-primary font-bold border-r-2 border-primary bg-surface-container'
                  : 'text-on-surface/60 hover:bg-surface-container hover:text-primary'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className="material-symbols-outlined text-[22px]"
                  style={isActive ? { fontVariationSettings: "'FILL' 1, 'wght' 500" } : {}}
                >
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User Profile */}
      <div className="px-3 pt-6 border-t border-white/5 relative">
        <div
          onClick={() => setProfileOpen(!profileOpen)}
          className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-surface-container transition-all duration-200 cursor-pointer"
        >
          <div className="w-9 h-9 shrink-0 rounded-full flex items-center justify-center text-xs font-bold bg-surface-container-highest text-primary border border-primary/20">
            SH
          </div>
          <div className="flex-1 overflow-hidden">
            <p className="text-sm font-semibold text-on-surface truncate">Sharvesh</p>
            <p className="text-[10px] uppercase tracking-widest text-primary/60">System Developer</p>
          </div>
          <span className={`material-symbols-outlined text-on-surface/40 text-sm transition-transform duration-200 ${profileOpen ? 'rotate-180' : ''}`}>
            expand_more
          </span>
        </div>

        {/* Dropdown popup */}
        {profileOpen && (
          <div className="absolute bottom-[72px] left-3 right-3 bg-surface-container-high border border-white/5 rounded-xl shadow-2xl overflow-hidden z-30" style={{ animation: 'slideUp 0.25s ease-out' }}>
            <div className="p-4 border-b border-white/5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-surface-container-highest flex items-center justify-center text-sm font-bold text-primary border border-primary/20">
                  SH
                </div>
                <div>
                  <p className="font-bold text-on-surface text-sm">Sharvesh</p>
                  <p className="text-[11px] text-on-surface-variant">ss1405@srmist.edu.in</p>
                </div>
              </div>
            </div>
            <div className="p-2">
              {[
                { icon: 'settings', label: 'Settings', sub: 'App preferences', onClick: () => {} },
                { icon: 'notifications', label: 'Notifications', sub: '3 unread', onClick: () => {} },
                { icon: isLight ? 'light_mode' : 'dark_mode', label: 'Theme', sub: isLight ? 'Light mode active' : 'Dark mode active', onClick: toggleTheme },
              ].map((item) => (
                <button
                  key={item.label}
                  onClick={item.onClick}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-surface-container transition-colors group"
                >
                  <span className="material-symbols-outlined text-on-surface-variant text-lg group-hover:text-primary transition-colors">{item.icon}</span>
                  <div>
                    <p className="text-sm text-on-surface">{item.label}</p>
                    <p className="text-[10px] text-on-surface-variant">{item.sub}</p>
                  </div>
                </button>
              ))}
            </div>
            <div className="p-2 border-t border-white/5">
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-error-container/20 transition-colors group">
                <span className="material-symbols-outlined text-error/50 text-lg">logout</span>
                <span className="text-sm text-error/70 group-hover:text-error">Sign Out</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

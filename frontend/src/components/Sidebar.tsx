import { useState } from 'react'
import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', icon: 'home', label: 'Home' },
  { to: '/route-planner', icon: 'directions_car', label: 'Route Planner' },
  { to: '/battery-analytics', icon: 'insert_chart', label: 'Battery Analytics' },
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
    <aside className="group/sidebar w-[260px] hidden md:flex flex-col h-full z-50 overflow-hidden shrink-0 bg-surface-100 border-r border-neon-cyan/8 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] absolute md:relative">
      {/* Logo */}
      <div className="p-6 pb-4">
        <div className="flex items-center gap-3 w-max">
          <div className="w-10 h-10  bg-gradient-to-br from-neon-cyan to-neon-green flex items-center justify-center glow-cyan shrink-0">
            <span className="material-symbols-outlined text-white text-lg">bolt</span>
          </div>
          <div className="opacity-100 transition-opacity duration-200 pointer-events-auto">
            <h1 className="font-headline font-bold text-base tracking-tight leading-none text-surface-900">
              IES_EV
            </h1>
            <p className="text-[10px] mt-0.5 text-surface-800/40 font-medium">
              Intelligent Management
            </p>
          </div>
        </div>
      </div>

      <div className="mx-5 mb-4 h-px bg-gradient-to-r from-transparent via-neon-cyan/10 to-transparent" />

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-1 stagger-children">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-4 px-4 py-3 mx-1 rounded-lg font-medium text-[15px] transition-all duration-300 overflow-hidden whitespace-nowrap ${
                isActive
                  ? 'bg-gradient-to-r from-neon-cyan/15 to-neon-green/5 text-neon-cyan border border-neon-cyan/15 shadow-[inset_0_0_10px_rgba(0,229,204,0.05)]'
                  : 'text-surface-800/50 hover:bg-surface-200/80 hover:text-surface-900'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={`material-symbols-outlined text-[22px] transition-all duration-300 ${isActive ? 'scale-110' : ''}`}
                  style={isActive ? { fontVariationSettings: "'FILL' 1, 'wght' 500" } : {}}
                >
                  {item.icon}
                </span>
                <span className="opacity-100 transition-opacity duration-200 delay-75">{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-neon-cyan animate-pulse opacity-100 transition-opacity" />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Interactive User Profile */}
      <div className="p-4 mt-auto relative">
        <div
          onClick={() => setProfileOpen(!profileOpen)}
          className=" p-2.5 md:p-3.5 flex items-center gap-3 bg-surface-200/50 border border-neon-cyan/8 cursor-pointer hover:border-neon-cyan/20 transition-all duration-300 group w-full min-w-[54px] md:min-w-0 mx-0 overflow-hidden"
        >
          <div className="w-9 h-9 shrink-0 rounded-full flex items-center justify-center text-xs font-bold bg-gradient-to-br from-neon-cyan/30 to-neon-green/20 text-neon-cyan border border-neon-cyan/20">
            SH
          </div>
          <div className="overflow-hidden flex-1 opacity-100 transition-opacity duration-200">
            <p className="text-sm font-semibold truncate text-surface-900">Sharvesh</p>
            <p className="text-[10px] text-surface-800/30 whitespace-nowrap">System Developer</p>
          </div>
          <span className={`material-symbols-outlined text-surface-800/30 text-sm opacity-100 transition-all duration-300 shrink-0 ${profileOpen ? 'rotate-180' : ''}`}>
            expand_more
          </span>
        </div>

        {/* Dropdown popup */}
        {profileOpen && (
          <div className="absolute bottom-20 left-4 right-4 glass-dark  shadow-2xl overflow-hidden z-30" style={{ animation: 'slideUp 0.25s ease-out' }}>
            <div className="p-4 border-b border-neon-cyan/8">
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-full bg-gradient-to-br from-neon-cyan/30 to-neon-green/20 flex items-center justify-center text-sm font-bold text-neon-cyan border border-neon-cyan/20">
                  SH
                </div>
                <div>
                  <p className="font-bold text-surface-900 text-sm">Sharvesh</p>
                  <p className="text-[11px] text-surface-800/40">ss1405@srmist.edu.in</p>
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
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-surface-200/30 transition-colors group"
                >
                  <span className="material-symbols-outlined text-surface-800/40 text-lg group-hover:text-neon-cyan transition-colors">{item.icon}</span>
                  <div>
                    <p className="text-sm text-surface-900">{item.label}</p>
                    <p className="text-[10px] text-surface-800/30">{item.sub}</p>
                  </div>
                </button>
              ))}
            </div>
            <div className="p-2 border-t border-neon-cyan/8">
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-neon-red/10 transition-colors group">
                <span className="material-symbols-outlined text-neon-red/50 text-lg">logout</span>
                <span className="text-sm text-neon-red/70 group-hover:text-neon-red">Sign Out</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', icon: 'home', label: 'Home' },
  { to: '/route-planner', icon: 'directions_car', label: 'Route Planner' },
  { to: '/battery-analytics', icon: 'insert_chart', label: 'Battery Analytics' },
  { to: '/charging-stations', icon: 'ev_station', label: 'Charging Stations' },
  { to: '/system-insights', icon: 'monitoring', label: 'System Insights' },
]

export default function Sidebar() {
  return (
    <aside className="w-[260px] flex flex-col h-full z-20 shrink-0 bg-ivory border-r border-brand-primary/10">
      {/* Logo */}
      <div className="p-7 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-brand-primary to-brand-secondary flex items-center justify-center glow-ice">
            <span className="material-symbols-outlined text-white text-xl">bolt</span>
          </div>
          <div>
            <h1 className="font-headline font-bold text-lg tracking-tight leading-none text-surface-900">
              IES_EV
            </h1>
            <p className="text-[11px] mt-0.5 text-surface-800/50 font-medium">
              Intelligent Management
            </p>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-6 mb-4">
        <div className="h-px bg-gradient-to-r from-transparent via-brand-primary/15 to-transparent" />
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-1.5 stagger-children">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3.5 px-4 py-3 rounded-xl font-medium text-[13px] transition-all duration-300 ${
                isActive
                  ? 'bg-brand-primary/10 text-brand-primary border border-brand-primary/15 shadow-sm'
                  : 'text-surface-800/60 hover:bg-ice/60 hover:text-surface-800'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={`material-symbols-outlined text-[20px] transition-all duration-300 ${isActive ? 'scale-110' : ''}`}
                  style={isActive ? { fontVariationSettings: "'FILL' 1, 'wght' 500" } : {}}
                >
                  {item.icon}
                </span>
                <span>{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-brand-primary animate-pulse" />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User Profile */}
      <div className="p-5 mt-auto">
        <div className="rounded-2xl p-4 flex items-center gap-3 bg-ice/50 border border-brand-primary/10">
          <div className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold bg-brand-primary/15 text-brand-primary">
            UN
          </div>
          <div className="overflow-hidden">
            <p className="text-sm font-semibold truncate text-surface-900">User Name</p>
            <p className="text-[11px] text-surface-800/40">Premium Plan</p>
          </div>
        </div>
      </div>
    </aside>
  )
}

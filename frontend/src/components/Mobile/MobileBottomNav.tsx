import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', icon: 'home', label: 'Home' },
  { to: '/route-planner', icon: 'directions_car', label: 'Route' },
  { to: '/battery-analytics', icon: 'insert_chart', label: 'Battery' },
  { to: '/charging-stations', icon: 'ev_station', label: 'Charging' },
  { to: '/system-insights', icon: 'monitoring', label: 'Insights' },
]

export default function MobileBottomNav() {
  return (
    <nav
      className="fixed bottom-0 left-0 right-0 h-16 z-[100] flex items-center justify-around bg-surface-100/90 backdrop-blur-xl border-t border-neon-cyan/10 md:hidden"
      style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
    >
      {navItems.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === '/'}
          aria-label={`Navigate to ${item.label}`}
          className={({ isActive }) =>
            `flex flex-col items-center justify-center gap-1 flex-1 min-h-[56px] relative transition-all duration-200 touch-manipulation ${
              isActive
                ? 'text-neon-cyan'
                : 'text-surface-800/50 active:scale-95'
            }`
          }
        >
          {({ isActive }) => (
            <>
              <span
                className={`material-symbols-outlined text-2xl transition-all duration-200 ${
                  isActive ? '-translate-y-0.5' : ''
                }`}
                style={isActive ? { fontVariationSettings: "'FILL' 1" } : {}}
              >
                {item.icon}
              </span>
              <span className="text-[10px] font-medium text-center whitespace-nowrap">
                {item.label}
              </span>
              {isActive && (
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-10 h-1 bg-gradient-to-r from-neon-cyan to-neon-green rounded-t" />
              )}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  )
}

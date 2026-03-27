import { Outlet, useLocation, NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import AnimatedBackground from './AnimatedBackground'

const navItems = [
  { to: '/', icon: 'home', label: 'Home' },
  { to: '/route-planner', icon: 'directions_car', label: 'Route' },
  { to: '/battery-analytics', icon: 'insert_chart', label: 'Battery' },
  { to: '/charging-stations', icon: 'ev_station', label: 'Stations' },
  { to: '/system-insights', icon: 'monitoring', label: 'Insights' },
]

export default function Layout() {
  const location = useLocation()

  return (
    <div className="flex h-[100dvh] w-full overflow-hidden bg-brand-bg relative">
      <a href="#main-content" className="absolute -top-12 left-0 bg-neon-cyan text-brand-bg px-4 py-2 rounded-br-lg z-[2000] focus:top-0 transition-all font-bold">
        Skip to main content
      </a>
      <AnimatedBackground />
      <Sidebar />
      <main id="main-content" className="flex-1 flex flex-col relative overflow-hidden pb-16 md:pb-0">
        <TopBar />
        <div className="flex-1 overflow-auto relative bg-grid">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
              className="h-full relative z-10"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      {/* Mobile Bottom Navigation (<768px) */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-surface-100/90 backdrop-blur-xl border-t border-neon-cyan/10 z-50 flex items-center justify-around px-2 pb-safe">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            aria-label={`Navigate to ${item.label}`}
            className={({ isActive }) =>
              `flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all ${
                isActive
                  ? 'text-neon-cyan glow-cyan'
                  : 'text-surface-800/60 hover:text-surface-900'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span className={`material-symbols-outlined text-2xl transition-transform ${isActive ? '-translate-y-1' : ''}`} style={isActive ? { fontVariationSettings: "'FILL' 1" } : {}}>
                  {item.icon}
                </span>
                {isActive && (
                  <span className="text-[9px] font-bold absolute bottom-2">{item.label}</span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  )
}

import { NavLink, useNavigate } from 'react-router-dom'
import { useVehicle } from '../../contexts/VehicleContext'
import { useCallback } from 'react'

const leftItems = [
  { to: '/route-planner',     icon: 'directions_car', label: 'Route'   },
]
const rightItems = [
  { to: '/battery-analytics', icon: 'insert_chart',   label: 'Battery' },
  { to: '/charging-stations', icon: 'ev_station',     label: 'Charging'},
]

function NavItem({ to, icon, label }: { to: string; icon: string; label: string }) {
  return (
    <NavLink
      to={to}
      aria-label={`Navigate to ${label}`}
      className={({ isActive }) =>
        `flex flex-col items-center justify-center gap-1 flex-1 min-h-[56px] relative transition-all duration-200 touch-manipulation ${
          isActive ? 'text-neon-cyan' : 'text-surface-800/50 active:scale-95'
        }`
      }
    >
      {({ isActive }) => (
        <>
          <span
            className={`material-symbols-outlined text-[22px] transition-all duration-200 ${isActive ? '-translate-y-0.5' : ''}`}
            style={isActive ? { fontVariationSettings: "'FILL' 1" } : {}}
          >
            {icon}
          </span>
          <span className="text-[9px] font-medium text-center whitespace-nowrap">{label}</span>
          {isActive && (
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-1 bg-gradient-to-r from-neon-cyan to-neon-green rounded-t" />
          )}
        </>
      )}
    </NavLink>
  )
}

export default function MobileBottomNav() {
  const navigate = useNavigate()
  const { currentVehicle, switchVehicle, setLabMinimized } = useVehicle()
  const isLabActive = currentVehicle === 'custom-lab'

  // Mirror desktop Sidebar behaviour: reset to default vehicle when leaving custom lab
  const handleHomePress = useCallback(() => {
    if (isLabActive) {
      switchVehicle('model-v-performance')
      setLabMinimized(false)
    }
  }, [isLabActive, switchVehicle, setLabMinimized])

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-[100] flex items-center bg-surface-100/90 backdrop-blur-xl border-t border-neon-cyan/10 md:hidden"
      style={{
        height: 'calc(4rem + env(safe-area-inset-bottom))',
        paddingBottom: 'env(safe-area-inset-bottom)',
      }}
    >
      {/* Home */}
      <NavLink
        to="/"
        end
        aria-label="Navigate to Home"
        onClick={handleHomePress}
        className={({ isActive }) =>
          `flex flex-col items-center justify-center gap-1 flex-1 min-h-[56px] relative transition-all duration-200 touch-manipulation ${
            isActive && !isLabActive ? 'text-neon-cyan' : 'text-surface-800/50 active:scale-95'
          }`
        }
      >
        {({ isActive }) => {
          const active = isActive && !isLabActive
          return (
            <>
              <span
                className={`material-symbols-outlined text-[22px] transition-all duration-200 ${active ? '-translate-y-0.5' : ''}`}
                style={active ? { fontVariationSettings: "'FILL' 1" } : {}}
              >
                home
              </span>
              <span className="text-[9px] font-medium text-center whitespace-nowrap">Home</span>
              {active && (
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-1 bg-gradient-to-r from-neon-cyan to-neon-green rounded-t" />
              )}
            </>
          )
        }}
      </NavLink>

      {/* Route */}
      {leftItems.map(item => <NavItem key={item.to} {...item} />)}

      {/* ── Centre Lab Button (raised) ── */}
      <div className="flex-1 flex flex-col items-center justify-end pb-2 relative">
        <button
          aria-label="Navigate to Custom Lab"
          onClick={() => {
            switchVehicle('custom-lab')
            setLabMinimized(false)
            navigate('/')
          }}
          className="flex flex-col items-center gap-1 touch-manipulation active:scale-95 transition-transform duration-150"
        >
          {/* Raised pill */}
          <div
            className="w-12 h-12 rounded-2xl flex items-center justify-center -translate-y-3 transition-all duration-200"
            style={{
              background: isLabActive
                ? 'linear-gradient(135deg, #A855F7, #7C3AED)'
                : 'linear-gradient(135deg, rgba(168,85,247,0.18), rgba(124,58,237,0.18))',
              border: '1.5px solid rgba(168,85,247,0.45)',
              boxShadow: isLabActive
                ? '0 0 22px rgba(168,85,247,0.55), 0 4px 14px rgba(0,0,0,0.35)'
                : '0 4px 12px rgba(168,85,247,0.22)',
            }}
          >
            <span
              className="material-symbols-outlined text-[22px] transition-all duration-200"
              style={{
                color: isLabActive ? '#fff' : '#A855F7',
                fontVariationSettings: isLabActive ? "'FILL' 1" : "'FILL' 0",
              }}
            >
              science
            </span>
          </div>
          <span
            className="text-[9px] font-medium whitespace-nowrap -mt-2"
            style={{ color: isLabActive ? '#A855F7' : undefined }}
          >
            Lab
          </span>
        </button>
        {isLabActive && (
          <div
            className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-1 rounded-t"
            style={{ background: 'linear-gradient(to right, #A855F7, #9333EA)' }}
          />
        )}
      </div>

      {/* Battery · Charging */}
      {rightItems.map(item => <NavItem key={item.to} {...item} />)}
    </nav>
  )
}

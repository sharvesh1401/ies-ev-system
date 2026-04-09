import { Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import AnimatedBackground from './AnimatedBackground'
import MobileTopBar from './Mobile/MobileTopBar'
import MobileBottomNav from './Mobile/MobileBottomNav'
import useWindowSize from '../hooks/useWindowSize'
import CustomVehicleEditor from './CustomVehicleEditor'
import { useVehicle } from '../contexts/VehicleContext'

export default function Layout() {
  const location = useLocation()
  const { isMobile } = useWindowSize()
  const { isCustomMode, isLabMinimized } = useVehicle()

  return (
    <div className="flex h-[100dvh] w-full overflow-hidden bg-background relative">
      <a href="#main-content" className="absolute -top-12 left-0 bg-primary text-on-primary px-4 py-2 rounded-br-lg z-[2000] focus:top-0 transition-all font-bold">
        Skip to main content
      </a>
      <AnimatedBackground />

      {/* Desktop Sidebar — hidden on mobile */}
      {!isMobile && <Sidebar />}

      {/* Mobile Top Bar — only on mobile */}
      {isMobile && <MobileTopBar />}

      <main
        id="main-content"
        className="flex-1 flex flex-col relative overflow-hidden transition-all duration-300 ease-out"
        style={{
          marginRight: isCustomMode && !isLabMinimized && !isMobile ? '320px' : '0px',
          ...(isMobile ? {
            paddingTop: 'calc(4rem + env(safe-area-inset-top))',
            paddingBottom: 'calc(4rem + env(safe-area-inset-bottom))',
          } : {}),
        }}
      >
        {/* Desktop Top Bar — hidden on mobile */}
        {!isMobile && <TopBar />}

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

      {/* Mobile Bottom Navigation — only on mobile */}
      {isMobile && <MobileBottomNav />}

      <CustomVehicleEditor />
    </div>
  )
}

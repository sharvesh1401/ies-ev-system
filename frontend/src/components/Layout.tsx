import { Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect } from 'react'
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

  const [showWakeup, setShowWakeup] = useState(false)
  useEffect(() => {
    const onWaking = () => setShowWakeup(true)
    const onReady  = () => setShowWakeup(false)
    window.addEventListener('backend:waking', onWaking)
    window.addEventListener('backend:ready',  onReady)
    return () => {
      window.removeEventListener('backend:waking', onWaking)
      window.removeEventListener('backend:ready',  onReady)
    }
  }, [])

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

      {/* Cold-start wakeup banner */}
      <AnimatePresence>
        {showWakeup && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 40 }}
            transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
            className="fixed bottom-20 left-1/2 -translate-x-1/2 z-[9999] pointer-events-none"
          >
            <div className="flex items-center gap-3 px-5 py-3 rounded-2xl bg-surface-container-highest/90 backdrop-blur-md border border-primary/20 shadow-2xl text-sm font-mono text-on-surface max-w-[90vw] sm:max-w-md">
              <span className="text-primary text-base shrink-0">⚡</span>
              <span>
                Backend waking up from sleep — first load may take up to{' '}
                <span className="text-primary font-bold">60 s</span>.
                Thanks for your patience!
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import AnimatedBackground from './AnimatedBackground'

export default function Layout() {
  const location = useLocation()

  return (
    <div className="flex h-screen w-full overflow-hidden bg-brand-bg">
      <AnimatedBackground />
      <Sidebar />
      <main className="flex-1 flex flex-col relative overflow-hidden">
        <TopBar />
        <div className="flex-1 overflow-auto relative bg-grid" key={location.pathname}>
          <div className="page-enter h-full relative z-10">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  )
}

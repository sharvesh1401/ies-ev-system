export default function RoutePlanner() {
  return (
    <div className="flex-1 flex overflow-hidden h-full">

      {/* ───── Sidebar: Planner Inputs ───── */}
      <section className="w-[400px] p-8 border-r border-surface-200 bg-ivory overflow-y-auto shrink-0">
        <div className="mb-8" style={{ animation: 'slideRight 0.4s ease-out' }}>
          <h3 className="text-2xl font-bold text-surface-900 mb-2">Configure Route</h3>
          <p className="text-surface-800/50 text-sm">Set your targets for intelligent energy optimization.</p>
        </div>

        <div className="space-y-6 stagger-children">
          {/* Destination */}
          <div>
            <label className="block text-[10px] font-bold text-surface-800/40 uppercase tracking-widest mb-2">Destination</label>
            <div className="relative">
              <span className="material-symbols-outlined absolute left-4 top-3 text-brand-primary">location_on</span>
              <input
                className="w-full bg-surface-50 border border-surface-200 text-surface-900 pl-12 pr-4 py-3 rounded-xl focus:ring-brand-primary focus:border-brand-primary outline-none transition-colors"
                placeholder="Enter address or landmark"
                defaultValue="Amsterdam, Netherlands"
                type="text"
              />
            </div>
          </div>

          {/* Distance & Duration */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-[10px] font-bold text-surface-800/40 uppercase tracking-widest mb-2">Distance Goal</label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-4 top-3 text-surface-800/30">route</span>
                <input
                  className="w-full bg-surface-50 border border-surface-200 text-surface-900 pl-12 pr-4 py-3 rounded-xl outline-none"
                  defaultValue="142 km"
                  type="text"
                />
              </div>
            </div>
            <div>
              <label className="block text-[10px] font-bold text-surface-800/40 uppercase tracking-widest mb-2">Duration</label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-4 top-3 text-surface-800/30">timer</span>
                <input
                  className="w-full bg-surface-50 border border-surface-200 text-surface-900 pl-12 pr-4 py-3 rounded-xl outline-none"
                  defaultValue="1h 45m"
                  type="text"
                />
              </div>
            </div>
          </div>

          {/* Efficiency Mode Info */}
          <div className="p-4 bg-ice/50 border border-brand-primary/15 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <span className="material-symbols-outlined text-brand-primary">info</span>
              <span className="text-sm font-semibold text-brand-primary">Efficiency Mode</span>
            </div>
            <p className="text-xs text-surface-800/60 leading-relaxed">
              System will prioritize routes with regenerative braking opportunities and lower average speed to conserve power.
            </p>
          </div>

          {/* Recalculate Button */}
          <button className="w-full bg-brand-primary hover:bg-brand-secondary text-white font-bold py-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 shadow-lg shadow-brand-primary/20">
            <span>Recalculate Route</span>
            <span className="material-symbols-outlined">auto_fix_high</span>
          </button>
        </div>

        {/* Historical Data */}
        <div className="mt-12" style={{ animation: 'fadeIn 0.8s ease-out' }}>
          <h4 className="text-[10px] font-bold text-surface-800/40 uppercase tracking-widest mb-4">Historical Data Comparison</h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-surface-800/50">Avg. Consumption</span>
              <span className="text-surface-900 font-mono font-bold">15.4 Wh/km</span>
            </div>
            <div className="w-full bg-surface-200 h-1.5 rounded-full overflow-hidden">
              <div className="bg-brand-primary h-full w-[65%] rounded-full" />
            </div>
          </div>
        </div>
      </section>

      {/* ───── Main: Map & Energy Overlay ───── */}
      <section className="flex-1 relative bg-ice">
        {/* Map background */}
        <div className="absolute inset-0 bg-ice-deep flex items-center justify-center">
          <div className="absolute inset-0 bg-gradient-to-br from-ice/40 to-ice-deep/80" />
          <span className="material-symbols-outlined text-8xl text-brand-primary/10 z-10">map</span>

          {/* Grid lines */}
          <svg className="absolute inset-0 w-full h-full opacity-5" xmlns="http://www.w3.org/2000/svg">
            {Array.from({ length: 20 }, (_, i) => (
              <line key={`h${i}`} x1="0" y1={`${i * 5}%`} x2="100%" y2={`${i * 5}%`} stroke="#5aa9e6" strokeWidth="0.5" />
            ))}
            {Array.from({ length: 20 }, (_, i) => (
              <line key={`v${i}`} x1={`${i * 5}%`} y1="0" x2={`${i * 5}%`} y2="100%" stroke="#5aa9e6" strokeWidth="0.5" />
            ))}
          </svg>
        </div>

        {/* SVG Route Line */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-10" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M 200 600 Q 400 400 600 300 T 900 100"
            fill="transparent"
            stroke="#5aa9e6"
            strokeWidth="4"
            strokeLinecap="round"
            opacity="0.6"
          />
          <circle cx="200" cy="600" r="8" fill="#5aa9e6" opacity="0.8" />
          <circle cx="900" cy="100" r="10" fill="#2ecc71" opacity="0.8" />
        </svg>

        {/* Energy Prediction Overlay */}
        <div className="absolute bottom-8 right-8 w-80 glass-ivory rounded-2xl p-6 shadow-2xl z-20" style={{ animation: 'slideUp 0.6s ease-out' }}>
          <div className="flex items-center justify-between mb-6">
            <h4 className="text-sm font-bold text-surface-800/60 uppercase tracking-widest">Energy Prediction</h4>
            <span className="px-2 py-1 bg-accent-success/15 text-accent-success text-[10px] rounded-md font-bold">OPTIMIZED</span>
          </div>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-end mb-1">
                <span className="text-xs text-surface-800/50">Energy Needed</span>
                <span className="text-xl font-bold text-surface-900">12.3 <span className="text-sm font-normal text-surface-800/40">kWh</span></span>
              </div>
              <div className="w-full bg-surface-200 h-1.5 rounded-full overflow-hidden">
                <div className="bg-brand-primary h-full w-[45%]" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-ice/50 rounded-xl border border-brand-primary/10">
                <p className="text-[10px] text-surface-800/40 uppercase mb-1">Arrival SoC</p>
                <p className="text-lg font-bold text-accent-success">64%</p>
              </div>
              <div className="p-3 bg-ice/50 rounded-xl border border-brand-primary/10">
                <p className="text-[10px] text-surface-800/40 uppercase mb-1">Confidence</p>
                <div className="flex items-center gap-2">
                  <p className="text-lg font-bold text-surface-900">87%</p>
                  <span className="material-symbols-outlined text-accent-success text-xs">verified</span>
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <button className="flex-1 text-xs font-bold text-surface-800 py-2.5 rounded-lg bg-surface-100 border border-surface-200 hover:border-brand-primary/20 transition-colors">Details</button>
              <button className="flex-1 text-xs font-bold text-white py-2.5 rounded-lg bg-brand-primary hover:bg-brand-secondary transition-colors">Start Navigation</button>
            </div>
          </div>
        </div>

        {/* Map Controls */}
        <div className="absolute top-8 right-8 flex flex-col gap-2 z-20">
          {['add', 'remove', 'layers'].map((icon) => (
            <button key={icon} className="w-10 h-10 glass-ivory flex items-center justify-center rounded-lg text-surface-800/60 hover:text-brand-primary transition-colors shadow-sm">
              <span className="material-symbols-outlined">{icon}</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}

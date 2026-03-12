export default function TopBar() {
  return (
    <header className="h-16 flex items-center justify-between px-8 z-10 shrink-0 bg-ivory/80 backdrop-blur-md border-b border-brand-primary/8">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-headline font-semibold text-surface-900">
          System Status
        </h2>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-accent-success/10 border border-accent-success/20">
          <div className="w-2 h-2 rounded-full bg-accent-success animate-pulse" />
          <span className="text-[10px] font-bold text-accent-success uppercase tracking-widest">Live</span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4 text-surface-800/40">
          <span className="material-symbols-outlined text-lg">wifi</span>
          <span className="material-symbols-outlined text-lg">schedule</span>
          <div className="flex items-center gap-2">
            <span
              className="material-symbols-outlined text-brand-primary text-lg"
              style={{ fontVariationSettings: "'FILL' 1" }}
            >
              battery_charging_full
            </span>
            <span className="text-sm font-bold text-surface-900">98%</span>
          </div>
        </div>
      </div>
    </header>
  )
}

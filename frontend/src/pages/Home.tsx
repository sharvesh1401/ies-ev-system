import CarModel from '../components/CarModel'

const efficiencyBars = [40, 55, 30, 65, 80, 45, 50, 35, 60, 90, 55, 40, 50, 65]

export default function Home() {
  return (
    <div className="flex-1 p-8 relative grid grid-cols-12 grid-rows-6 gap-6 overflow-hidden h-full">

      {/* ───── Central Zone: 3D Car ───── */}
      <div className="col-span-12 lg:col-span-6 row-span-4 relative flex items-center justify-center">
        <CarModel />
      </div>

      {/* ───── Left Panel: Vehicle Overview ───── */}
      <div className="col-span-12 lg:col-span-3 row-span-4 flex flex-col gap-5 stagger-children">
        {/* SoC Card */}
        <div className="glass-ivory p-6 rounded-2xl flex-1 flex flex-col justify-between card-hover border-l-4 border-l-brand-primary">
          <div>
            <p className="text-[10px] font-bold text-surface-800/40 uppercase tracking-widest mb-1">State of Charge</p>
            <div className="flex items-baseline gap-2">
              <h3 className="text-5xl font-headline font-bold text-surface-900 tracking-tighter">76%</h3>
              <span
                className="material-symbols-outlined text-brand-primary animate-pulse"
                style={{ fontVariationSettings: "'FILL' 1" }}
              >
                bolt
              </span>
            </div>
          </div>
          <div className="space-y-3 mt-4">
            {[
              ['Health (SoH)', '94%', 'text-accent-success'],
              ['Estimated Range', '312 km', 'text-surface-900'],
              ['Battery Temp', '29°C', 'text-surface-900'],
            ].map(([label, value, color]) => (
              <div key={label} className="flex justify-between items-center border-b border-surface-200 pb-2">
                <span className="text-surface-800/50 text-sm">{label}</span>
                <span className={`${color} font-mono font-bold`}>{value}</span>
              </div>
            ))}
            <div className="flex justify-between items-center pt-1">
              <span className="text-surface-800/50 text-sm">Current Power</span>
              <span className="text-brand-primary font-mono font-bold">4.3 kW</span>
            </div>
          </div>
        </div>

        {/* Sentry Mode */}
        <div className="glass-ivory p-5 rounded-2xl flex items-center gap-4 group card-hover cursor-pointer">
          <div className="w-12 h-12 rounded-xl bg-brand-secondary/10 flex items-center justify-center text-brand-secondary">
            <span className="material-symbols-outlined">security</span>
          </div>
          <div>
            <h4 className="text-sm font-bold text-surface-900">Sentry Mode</h4>
            <p className="text-[11px] text-surface-800/40">Active • 2 Events Logged</p>
          </div>
        </div>
      </div>

      {/* ───── Right Panel: Map / Charger ───── */}
      <div className="col-span-12 lg:col-span-3 row-span-4 flex flex-col gap-5 stagger-children">
        <div className="glass-ivory rounded-2xl flex-1 overflow-hidden relative card-hover">
          {/* Nearby Charger Label */}
          <div className="absolute top-4 left-4 z-10 glass-ivory p-3 rounded-xl shadow-md">
            <h4 className="text-[10px] font-bold text-surface-800/50 mb-1 uppercase tracking-widest">Nearby Charger</h4>
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-brand-primary text-sm">ev_station</span>
              <span className="text-sm font-semibold text-surface-900">Amsterdam CS</span>
            </div>
          </div>

          {/* Map placeholder */}
          <div className="h-full w-full bg-ice relative flex items-center justify-center">
            <div className="absolute inset-0 bg-gradient-to-b from-ice/50 to-ice-deep/50" />
            <span className="material-symbols-outlined text-6xl text-brand-primary/20 z-10">map</span>

            {/* Destination overlay */}
            <div className="absolute bottom-4 left-4 right-4 glass-ivory p-4 rounded-xl z-10 shadow-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] font-bold text-brand-primary uppercase tracking-widest">Recommended</span>
                <span className="text-xs font-mono text-surface-800/50">0.8km away</span>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-sm font-bold text-surface-900">FastNed Amsterdam</p>
                <span className="material-symbols-outlined text-brand-primary hover:text-brand-secondary cursor-pointer transition-colors">near_me</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ───── Bottom: Energy Efficiency Graph ───── */}
      <div className="col-span-12 row-span-2 glass-ivory rounded-2xl p-6 relative flex flex-col card-hover">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-bold text-surface-900 tracking-wide uppercase flex items-center gap-2">
            <span className="material-symbols-outlined text-brand-primary text-lg">insights</span>
            Energy Efficiency
          </h3>
          <div className="flex gap-2">
            {['1H', '6H', '24H'].map((t) => (
              <button
                key={t}
                className={`px-3 py-1 text-[10px] font-bold rounded-full border transition-colors ${
                  t === '6H'
                    ? 'bg-brand-primary/15 border-brand-primary/20 text-brand-primary'
                    : 'bg-surface-100 border-surface-200 text-surface-800/40 hover:border-brand-primary/20'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 flex items-end justify-between gap-1.5 pb-2">
          {efficiencyBars.map((h, i) => (
            <div
              key={i}
              className={`flex-1 rounded-t-md transition-all duration-300 hover:opacity-80 ${
                h >= 80
                  ? 'bg-gradient-to-t from-brand-secondary/40 to-brand-secondary/10'
                  : 'bg-gradient-to-t from-brand-primary/30 to-brand-primary/8'
              }`}
              style={{ height: `${h}%` }}
            />
          ))}
        </div>

        <div className="flex justify-between mt-3">
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
              <span className="text-[10px] text-surface-800/40 font-bold uppercase">Average</span>
              <span className="text-sm font-mono font-bold text-surface-900">142 Wh/km</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-surface-800/40 font-bold uppercase">Optimal</span>
              <span className="text-sm font-mono font-bold text-brand-secondary">118 Wh/km</span>
            </div>
          </div>
          <div className="text-[10px] text-surface-800/30 self-end font-mono">UPDATED: 2 SEC AGO</div>
        </div>
      </div>

      {/* ───── Background glows ───── */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-brand-primary/5 rounded-full blur-[120px] -z-10 translate-x-1/2 -translate-y-1/2 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-ivory-warm/60 rounded-full blur-[150px] -z-10 -translate-x-1/4 translate-y-1/4 pointer-events-none" />
    </div>
  )
}

const cellGrid = Array.from({ length: 24 }, (_, i) => (i === 6 ? 'warning' : 'ok'))

export default function BatteryAnalytics() {
  return (
    <div className="p-8 max-w-7xl mx-auto overflow-y-auto">

      {/* ───── Summary Stats ───── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 stagger-children">
        <StatCard
          icon="favorite"
          label="Battery Health"
          value="94%"
          badge={<span className="text-accent-success text-sm font-bold flex items-center"><span className="material-symbols-outlined text-sm">arrow_upward</span>Nominal</span>}
          bar={94}
        />
        <StatCard
          icon="cycle"
          label="Charge Cycles"
          value="382"
          sub="/ 2000 Peak"
          note="Optimized charging pattern active"
        />
        <StatCard
          icon="hourglass_empty"
          label="Lifetime Remaining"
          value="6.1"
          sub="years"
          note="Based on current degradation rate"
        />
      </div>

      {/* ───── Degradation Graph ───── */}
      <div className="glass-ivory p-8 rounded-2xl mb-8 card-hover">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h3 className="text-lg font-bold text-surface-900">Detailed Degradation Model</h3>
            <p className="text-sm text-surface-800/50">State of Health (SoH) projection vs. actual performance</p>
          </div>
          <div className="flex bg-surface-100 p-1 rounded-lg">
            {['6 Months', '1 Year', 'All Time'].map((t, i) => (
              <button
                key={t}
                className={`px-4 py-1.5 text-xs font-bold rounded-md transition-colors ${i === 0 ? 'bg-white shadow-sm text-surface-900' : 'text-surface-800/40 hover:text-surface-900'}`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <div className="h-64 relative">
          <svg className="w-full h-full overflow-visible" viewBox="0 0 1000 200" preserveAspectRatio="none">
            <defs>
              <linearGradient id="lineGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{ stopColor: 'rgba(90, 169, 230, 0.2)' }} />
                <stop offset="100%" style={{ stopColor: 'rgba(90, 169, 230, 0)' }} />
              </linearGradient>
            </defs>
            <path d="M0,40 Q150,35 300,50 T600,65 T1000,80 L1000,200 L0,200 Z" fill="url(#lineGrad)" />
            <path d="M0,40 Q150,35 300,50 T600,65 T1000,80" fill="none" stroke="#5aa9e6" strokeWidth="3" />
            <circle cx="300" cy="50" r="4" fill="white" stroke="#5aa9e6" strokeWidth="2" />
            <circle cx="600" cy="65" r="4" fill="white" stroke="#5aa9e6" strokeWidth="2" />
          </svg>
          {/* Grid lines */}
          <div className="absolute inset-0 flex flex-col justify-between opacity-10 pointer-events-none">
            {[0, 1, 2, 3].map((i) => <div key={i} className="border-t border-surface-800/10 w-full" />)}
          </div>
          {/* Month labels */}
          <div className="absolute bottom-[-24px] w-full flex justify-between text-[10px] font-bold text-surface-800/40 uppercase tracking-widest">
            {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'].map((m) => <span key={m}>{m}</span>)}
          </div>
        </div>
      </div>

      {/* ───── Bento Grid ───── */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Cell Voltage */}
        <div className="lg:col-span-2 glass-ivory p-6 rounded-2xl card-hover">
          <div className="flex items-center justify-between mb-6">
            <h4 className="font-bold text-surface-900">Cell Voltage Variance</h4>
            <span className="text-xs bg-ice px-2 py-1 rounded text-brand-primary">96 Cells Active</span>
          </div>
          <div className="grid grid-cols-12 gap-1.5">
            {cellGrid.map((status, i) => (
              <div
                key={i}
                className={`h-8 border rounded-sm ${
                  status === 'warning'
                    ? 'bg-accent-warning/20 border-accent-warning/30'
                    : 'bg-accent-success/20 border-accent-success/30'
                }`}
              />
            ))}
          </div>
          <div className="mt-4 flex items-center justify-between text-xs text-surface-800/50">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-accent-success" /> Balanced</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-accent-warning" /> Deviation</span>
            </div>
            <span className="font-bold">Δ 0.002V</span>
          </div>
        </div>

        {/* Thermal Gauge */}
        <div className="bg-surface-900 text-white p-6 rounded-2xl shadow-lg relative overflow-hidden card-hover">
          <div className="relative z-10">
            <h4 className="font-bold mb-4">Core Temperature</h4>
            <div className="flex items-center justify-center py-4">
              <div className="relative w-32 h-32">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="10" />
                  <circle cx="50" cy="50" r="45" fill="none" stroke="white" strokeWidth="10" strokeDasharray="198" strokeDashoffset="100" />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-2xl font-bold">34°C</span>
                  <span className="text-[10px] uppercase font-bold text-sky-200">Stable</span>
                </div>
              </div>
            </div>
            <div className="flex justify-between text-xs mt-2 border-t border-white/10 pt-4">
              <span className="text-sky-200">Liquid Cooling</span>
              <span className="text-accent-success font-bold">ACTIVE</span>
            </div>
          </div>
        </div>

        {/* Efficiency Metrics */}
        <div className="glass-ivory p-6 rounded-2xl flex flex-col justify-between card-hover">
          <div>
            <h4 className="font-bold text-surface-900 mb-1">Energy Efficiency</h4>
            <p className="text-xs text-surface-800/40 mb-4">Last 30 days avg</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-end justify-between">
              <span className="text-xs font-bold text-surface-800/40">WH/MI</span>
              <span className="text-xl font-bold text-surface-900">285</span>
            </div>
            <div className="flex items-end justify-between">
              <span className="text-xs font-bold text-surface-800/40">REGEN</span>
              <span className="text-xl font-bold text-surface-900">12%</span>
            </div>
          </div>
          <button className="mt-4 w-full py-2.5 bg-brand-primary/10 border border-brand-primary/15 rounded-lg text-xs font-bold hover:bg-brand-primary/20 transition-colors text-brand-primary">
            FULL REPORT
          </button>
        </div>
      </div>

      {/* ───── Maintenance ───── */}
      <div className="mt-8 flex flex-col md:flex-row gap-6">
        <div className="flex-1 glass-ivory p-6 rounded-2xl flex items-center gap-6 card-hover">
          <div className="w-16 h-16 bg-ice rounded-full flex items-center justify-center text-brand-primary shrink-0">
            <span className="material-symbols-outlined text-3xl">verified_user</span>
          </div>
          <div>
            <h5 className="font-bold text-surface-900">Maintenance Recommendation</h5>
            <p className="text-sm text-surface-800/50">
              System predicts next deep cycle calibration in <span className="text-brand-primary font-bold">124 days</span>. No immediate service required.
            </p>
          </div>
          <div className="ml-auto shrink-0">
            <button className="px-6 py-2.5 bg-brand-primary text-white rounded-xl text-sm font-bold shadow-md hover:bg-brand-secondary transition-colors">Schedule Early</button>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── Helper component ── */
function StatCard({
  icon, label, value, sub, badge, bar, note,
}: {
  icon: string; label: string; value: string; sub?: string;
  badge?: React.ReactNode; bar?: number; note?: string;
}) {
  return (
    <div className="glass-ivory p-6 rounded-2xl relative overflow-hidden card-hover">
      <div className="absolute top-0 right-0 p-4 opacity-5">
        <span className="material-symbols-outlined text-6xl">{icon}</span>
      </div>
      <p className="text-sm text-surface-800/50 font-medium mb-1">{label}</p>
      <div className="flex items-baseline gap-2">
        <h3 className="text-4xl font-bold text-surface-900">{value}</h3>
        {sub && <span className="text-surface-800/40 text-sm font-medium">{sub}</span>}
        {badge}
      </div>
      {bar !== undefined && (
        <div className="mt-4 w-full bg-surface-200 h-2 rounded-full overflow-hidden">
          <div className="bg-accent-success h-full" style={{ width: `${bar}%` }} />
        </div>
      )}
      {note && <p className="mt-4 text-xs text-surface-800/40">{note}</p>}
    </div>
  )
}

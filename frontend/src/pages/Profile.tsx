export default function Profile() {
  return (
    <div className="p-6 md:p-8 max-w-4xl mx-auto overflow-y-auto space-y-6 no-scrollbar">

      {/* ═══ Header Card ═══ */}
      <div className="glass-dark p-6 md:p-8 card-hover relative overflow-hidden border border-neon-cyan/20">
        <div className="absolute -top-16 -right-16 w-48 h-48 bg-neon-cyan/10 rounded-full blur-[60px]" />
        <div className="flex items-center gap-5 relative z-10">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-neon-cyan to-neon-green flex items-center justify-center shadow-[0_0_30px_rgba(0,229,204,0.4)] shrink-0">
            <span className="text-4xl">⚡</span>
          </div>
          <div>
            <h1 className="text-2xl font-headline font-bold text-surface-900 tracking-tight">Sharvesh Selvakumar</h1>
            <div className="flex flex-wrap gap-2 mt-2">
              <span className="px-3 py-1 text-[10px] font-mono font-bold uppercase tracking-widest bg-gradient-to-r from-neon-cyan/20 to-neon-green/10 text-neon-cyan border border-neon-cyan/30 shadow-[0_0_10px_rgba(0,229,204,0.2)] rounded-full">
                System Developer
              </span>
              <span className="px-3 py-1 text-[10px] font-mono font-bold uppercase tracking-widest border border-neon-purple/30 text-neon-purple bg-neon-purple/10 rounded-full">
                ML Engineer
              </span>
            </div>
            <p className="text-sm text-surface-800/60 mt-2 font-medium">@ SRM Institute of Science & Technology</p>
          </div>
        </div>
      </div>

      {/* ═══ Quick Links ═══ */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 stagger-children">
        {[
          { icon: 'code', title: 'GitHub', subtitle: '@sharvesh1401', href: 'https://github.com/sharvesh1401', stat: '15 repositories', color: 'neon-cyan' },
          { icon: 'work', title: 'Portfolio', subtitle: 'sharveshportfolio.com', href: 'https://sharveshportfolio.com', stat: '5 major projects', color: 'neon-green' },
          { icon: 'group', title: 'LinkedIn', subtitle: 'Connect', href: 'https://linkedin.com/in/sharvesh-selvakumar', stat: '500+ connections', color: 'neon-purple' },
        ].map((link) => (
          <a
            key={link.title}
            href={link.href}
            target="_blank"
            rel="noopener noreferrer"
            className="glass-dark p-5 card-hover border border-neon-cyan/10 group block"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-10 h-10 rounded-full bg-${link.color}/10 border border-${link.color}/20 flex items-center justify-center text-${link.color} group-hover:shadow-[0_0_15px_currentColor] transition-shadow`}>
                <span className="material-symbols-outlined text-lg">{link.icon}</span>
              </div>
              <div>
                <p className="text-sm font-bold text-surface-900">{link.title}</p>
                <p className="text-[10px] text-surface-800/40 font-mono">{link.subtitle}</p>
              </div>
            </div>
            <p className="text-[10px] font-mono text-surface-800/40 uppercase tracking-widest">{link.stat}</p>
          </a>
        ))}
      </div>

      {/* ═══ About ═══ */}
      <div className="glass-dark p-6 card-hover border border-neon-cyan/10">
        <h3 className="font-headline font-bold text-surface-900 tracking-tight mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-neon-cyan text-lg">info</span>
          About the Developer
        </h3>
        <p className="text-sm text-surface-800/70 leading-relaxed mb-3">
          Built <strong className="text-surface-900">IES_EV</strong> — a hybrid ML+Physics system for electric vehicle energy management that combines transformer-based
          neural networks with physics validation. Published research achieving
          <strong className="text-neon-cyan"> 2.1% MAPE</strong> (4× better than industry standards).
        </p>
        <p className="text-sm text-surface-800/70 leading-relaxed">
          Specializes in <strong className="text-surface-900">edge AI deployment</strong>, real-time systems, and battery analytics.
          This platform demonstrates production-ready ML inference running on constrained hardware with sub-second latency.
        </p>
      </div>

      {/* ═══ Skills & Technologies ═══ */}
      <div className="glass-dark p-6 card-hover border border-neon-cyan/10">
        <h3 className="font-headline font-bold text-surface-900 tracking-tight mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-neon-cyan text-lg">build</span>
          Skills & Technologies
        </h3>
        <div className="flex flex-wrap gap-2">
          {[
            { name: 'Python', level: 'expert' },
            { name: 'PyTorch', level: 'expert' },
            { name: 'Transformers', level: 'expert' },
            { name: 'XGBoost', level: 'expert' },
            { name: 'React', level: 'advanced' },
            { name: 'TypeScript', level: 'advanced' },
            { name: 'Node.js', level: 'advanced' },
            { name: 'FastAPI', level: 'advanced' },
            { name: 'Docker', level: 'intermediate' },
            { name: 'Mapbox', level: 'intermediate' },
          ].map((skill) => (
            <span
              key={skill.name}
              className={`px-4 py-2 text-sm font-medium border transition-all hover:scale-105 cursor-default rounded-full ${
                skill.level === 'expert'
                  ? 'bg-accent-success/10 border-accent-success/30 text-accent-success shadow-[0_0_8px_rgba(0,230,118,0.15)]'
                  : skill.level === 'advanced'
                  ? 'bg-neon-cyan/10 border-neon-cyan/30 text-neon-cyan shadow-[0_0_8px_rgba(0,229,204,0.15)]'
                  : 'bg-surface-200/50 border-surface-300/50 text-surface-800'
              }`}
            >
              {skill.name}
            </span>
          ))}
        </div>
      </div>

      {/* ═══ Research & Publications ═══ */}
      <div className="glass-dark p-6 card-hover border border-neon-cyan/10">
        <h3 className="font-headline font-bold text-surface-900 tracking-tight mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-neon-cyan text-lg">science</span>
          Research & Publications
        </h3>
        <div className="bg-surface-200/30 border border-neon-cyan/10 p-5 group hover:border-neon-cyan/20 transition-colors rounded-xl">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h4 className="text-sm font-bold text-surface-900 group-hover:text-neon-cyan transition-colors leading-snug">
                Development of Intelligent Energy & Routing System for On-board Electric Vehicles
              </h4>
              <p className="text-[11px] text-surface-800/50 mt-2 font-mono">IEEE Conference • 2026</p>
            </div>
            <span className="px-3 py-1 text-[9px] font-mono font-bold uppercase tracking-widest bg-neon-yellow/10 border border-neon-yellow/30 text-neon-yellow shrink-0 shadow-[0_0_10px_rgba(255,214,0,0.15)] rounded-full">
              Submitted
            </span>
          </div>
        </div>
      </div>

      {/* ═══ Contact ═══ */}
      <div className="glass-dark p-6 card-hover border border-neon-cyan/10">
        <h3 className="font-headline font-bold text-surface-900 tracking-tight mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-neon-cyan text-lg">contact_mail</span>
          Contact
        </h3>
        <div className="space-y-3">
          {[
            { icon: 'mail', text: 'ss1405@srmist.edu.in' },
            { icon: 'phone', text: '+91 XXXXX XXXXX' },
            { icon: 'location_on', text: 'Chennai, Tamil Nadu, India' },
          ].map((item) => (
            <div key={item.icon} className="flex items-center gap-3 group">
              <div className="w-9 h-9 rounded-full bg-surface-200/50 border border-surface-300/30 flex items-center justify-center text-surface-800/50 group-hover:text-neon-cyan group-hover:border-neon-cyan/30 transition-colors">
                <span className="material-symbols-outlined text-base">{item.icon}</span>
              </div>
              <span className="text-sm text-surface-800/70 font-mono group-hover:text-surface-900 transition-colors">{item.text}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  )
}

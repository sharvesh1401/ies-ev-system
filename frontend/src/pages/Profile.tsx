import { useState } from 'react'

export default function Profile() {
  const [isEditing, setIsEditing] = useState(false)
  const [profile, setProfile] = useState({
    name: 'Sharvesh',
    role: 'System Developer',
    email: 's_sharvesh@outlook.com',
    location: 'Chennai, India',
    timezone: 'IST (UTC +5:30)',
    focus: 'EV Telemetry, RTOS Systems',
    joinDate: 'October 2023',
    website: 'meridian-ev.io',
    github: 'github.com/sharvesh',
    repo: 'github.com/sharvesh/meridian-ies-ev',
  })
  const [draft, setDraft] = useState({ ...profile })

  const handleSave = () => { setProfile({ ...draft }); setIsEditing(false) }
  const handleCancel = () => { setDraft({ ...profile }); setIsEditing(false) }

  const initials = profile.name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase()

  return (
    <div className="w-full">
      <div className="max-w-[680px] mx-auto px-6 pt-10 pb-16 flex flex-col gap-10">

        {/* ── Header Card ── */}
        <div className="bg-surface-container-highest rounded-2xl p-7 flex items-start gap-6">

          {/* Avatar + ONLINE badge */}
          <div className="relative shrink-0">
            <div className="w-[96px] h-[96px] rounded-2xl bg-white flex items-center justify-center shadow-sm border border-outline-variant/20">
              <span className="text-[30px] font-black text-primary leading-none">{initials}</span>
            </div>
            <div className="absolute -bottom-3 left-3 flex items-center gap-1.5 bg-[#00C853] px-3 py-[5px] rounded-full shadow-sm z-10">
              <span className="w-[6px] h-[6px] rounded-full bg-white animate-pulse shrink-0" />
              <span className="text-[9px] font-black text-white uppercase tracking-[0.15em]">Online</span>
            </div>
          </div>

          {/* Name / role / email */}
          <div className="flex-1 min-w-0 pt-1">
            {isEditing ? (
              <div className="space-y-2">
                <input
                  value={draft.name}
                  onChange={e => setDraft({ ...draft, name: e.target.value })}
                  className="text-[22px] font-black text-on-surface bg-surface-container border border-outline-variant/30 rounded-xl px-3 py-1.5 outline-none focus:border-primary/50 w-full"
                />
                <input
                  value={draft.role}
                  onChange={e => setDraft({ ...draft, role: e.target.value })}
                  className="text-[11px] font-mono font-black uppercase tracking-[0.2em] text-primary bg-surface-container border border-outline-variant/30 rounded-xl px-3 py-1.5 outline-none focus:border-primary/50 w-full"
                />
                <input
                  value={draft.email}
                  onChange={e => setDraft({ ...draft, email: e.target.value })}
                  className="text-[13px] text-on-surface-variant bg-surface-container border border-outline-variant/30 rounded-xl px-3 py-1.5 outline-none focus:border-primary/50 w-full"
                />
              </div>
            ) : (
              <>
                <h1 className="text-[22px] font-black text-on-surface leading-tight">{profile.name}</h1>
                <p className="text-[11px] font-mono font-bold text-primary uppercase tracking-[0.2em] mt-1.5">{profile.role}</p>
                <p className="text-[13px] text-on-surface-variant mt-2">{profile.email}</p>
              </>
            )}
          </div>

          {/* Edit / Save buttons — top-right aligned */}
          <div className="shrink-0 pt-1">
            {isEditing ? (
              <div className="flex gap-2">
                <button
                  onClick={handleCancel}
                  className="px-4 py-2 rounded-xl border border-outline-variant/40 text-[13px] font-semibold text-on-surface hover:bg-black/5 dark:hover:bg-white/5 transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  className="px-4 py-2 rounded-xl bg-primary text-on-primary text-[13px] font-semibold hover:brightness-110 transition-all"
                >
                  Save
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsEditing(true)}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-outline-variant/40 text-[13px] font-semibold text-on-surface hover:bg-black/5 dark:hover:bg-white/5 transition-all"
              >
                <span className="material-symbols-outlined text-[16px]">edit</span>
                Edit Profile
              </button>
            )}
          </div>
        </div>

        {/* ── Personal Information ── */}
        <div className="flex flex-col gap-5">
          {/* Section header */}
          <div className="flex items-center gap-5">
            <span className="text-[10px] font-mono font-bold text-on-surface-variant uppercase tracking-[0.2em] shrink-0">
              Personal Information
            </span>
            <div className="flex-1 h-px bg-outline-variant" />
          </div>

          {/* 2×2 grid */}
          <div className="grid grid-cols-2 gap-4">
            {([
              { label: 'Base Location', key: 'location' as const },
              { label: 'Timezone',      key: 'timezone'  as const },
              { label: 'Focus Areas',  key: 'focus'     as const },
              { label: 'Join Date',    key: 'joinDate'  as const },
            ]).map(item => (
              <div
                key={item.label}
                className="bg-surface-container-high rounded-2xl px-5 py-4 min-h-[84px] flex flex-col justify-center"
              >
                <p className="text-[9px] font-mono font-bold text-on-surface-variant uppercase tracking-[0.15em] mb-2">
                  {item.label}
                </p>
                {isEditing ? (
                  <input
                    value={draft[item.key]}
                    onChange={e => setDraft({ ...draft, [item.key]: e.target.value })}
                    className="text-[15px] font-medium text-on-surface bg-surface-container border border-outline-variant/30 rounded-lg px-2 py-1 outline-none focus:border-primary/50 w-full"
                  />
                ) : (
                  <p className="text-[15px] font-medium text-on-surface">{profile[item.key]}</p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ── Professional Links ── */}
        <div className="flex flex-col gap-5">
          {/* Section header */}
          <div className="flex items-center gap-5">
            <span className="text-[10px] font-mono font-bold text-on-surface-variant uppercase tracking-[0.2em] shrink-0">
              Professional Links
            </span>
            <div className="flex-1 h-px bg-outline-variant" />
          </div>

          {/* Link rows */}
          <div className="flex flex-col gap-3">
            {([
              {
                icon: 'language',
                label: 'Website',
                key: 'website' as const,
                href: 'https://meridian-ev.io',
                iconBg: 'bg-primary/15',
                iconColor: 'text-primary',
              },
              {
                icon: 'laptop_mac',
                label: 'GitHub',
                key: 'github' as const,
                href: 'https://github.com/sharvesh1401',
                iconBg: 'bg-on-surface/8',
                iconColor: 'text-on-surface-variant',
              },
              {
                icon: 'account_tree',
                label: 'GitHub Repository',
                key: 'repo' as const,
                href: 'https://github.com/sharvesh1401/ies-ev-system',
                iconBg: 'bg-secondary-container/15',
                iconColor: 'text-secondary-container',
              },
            ]).map(link => (
              <div
                key={link.label}
                className="flex items-center gap-5 px-5 py-4 bg-surface-container-high rounded-2xl"
              >
                {/* Icon circle */}
                <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${link.iconBg} ${link.iconColor}`}>
                  <span
                    className="material-symbols-outlined text-[20px]"
                    style={{ fontVariationSettings: "'FILL' 1" }}
                  >
                    {link.icon}
                  </span>
                </div>

                {/* Label + URL */}
                <div className="flex-1 min-w-0">
                  <p className="text-[15px] font-semibold text-on-surface leading-tight">{link.label}</p>
                  {isEditing ? (
                    <input
                      value={draft[link.key]}
                      onChange={e => setDraft({ ...draft, [link.key]: e.target.value })}
                      className="text-[11px] font-mono text-on-surface-variant bg-surface-container border border-outline-variant/30 rounded-lg px-2 py-0.5 mt-1 outline-none focus:border-primary/50 w-full"
                    />
                  ) : (
                    <p className="text-[11px] font-mono text-on-surface-variant mt-0.5 truncate">{profile[link.key]}</p>
                  )}
                </div>

                {/* Arrow */}
                {!isEditing && (
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-8 h-8 flex items-center justify-center shrink-0 text-primary hover:text-primary/70 transition-colors"
                  >
                    <span className="material-symbols-outlined text-[20px]">north_east</span>
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ── Footer status bar ── */}
        <div className="flex items-center justify-between pt-6 border-t border-outline-variant">
          <div className="flex items-center gap-2">
            <span className="w-[7px] h-[7px] rounded-full bg-[#00C853] animate-pulse shrink-0" />
            <span className="text-[9px] font-mono text-on-surface-variant uppercase tracking-[0.15em]">
              Secure Handshake: Meridian-AES-256
            </span>
          </div>
          <span className="text-[9px] font-mono text-on-surface-variant uppercase tracking-[0.15em]">
            System Node ID: 882-XQ-Profile
          </span>
        </div>

      </div>
    </div>
  )
}

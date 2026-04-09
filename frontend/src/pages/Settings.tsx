import { useState } from 'react'
import { useVehicle } from '../contexts/VehicleContext'

type TabType = 'General' | 'Vehicle' | 'Navigation' | 'AI' | 'Charging' | 'Account' | 'Advanced'

export default function Settings() {
  const [activeTab, setActiveTab] = useState<TabType>('General')
  const { isCustomMode } = useVehicle()

  const [settings, setSettings] = useState({
    // General
    language: 'English (US)',
    units: 'Metric (km, °C)',
    notifications: true,
    compactDensity: false,
    reduceMotion: false,
    
    // Vehicle
    predictionMode: 'Balanced',
    routeOptimization: 'Energy-efficient',
    confidenceThreshold: 85,
    
    // Navigation
    avoidTolls: false,
    avoidHighways: false,
    terrainAware: true,
    fastChargingOnly: true,
    
    // Charging
    minBatteryReserve: 15,
    targetSoC: 80,
    costVsSpeed: 50, // 0 = Cheap, 100 = Fast
    
    // Account
    dataSync: true,
    
    // Advanced
    livePredictions: false,
    logApiResp: false,
  })

  const updateSetting = (key: string, value: any) => {
    setSettings((s) => ({ ...s, [key]: value }))
  }

  const tabList: { id: TabType; icon: string; label: string }[] = [
    { id: 'General', icon: 'settings', label: 'General Preferences' },
    { id: 'Vehicle', icon: 'directions_car', label: 'Vehicle Configuration' },
    { id: 'Navigation', icon: 'explore', label: 'Navigation & Routing' },
    { id: 'AI', icon: 'neurology', label: 'AI & ML Tuning' },
    { id: 'Charging', icon: 'ev_station', label: 'Charging Parameters' },
    { id: 'Account', icon: 'person', label: 'Account & Data' },
    { id: 'Advanced', icon: 'code', label: 'Developer / Debug' },
  ]

  return (
    <div className="w-full h-full flex flex-col md:flex-row overflow-hidden bg-background">
      {/* Sidebar Navigation */}
      <div className="w-full md:w-[280px] shrink-0 bg-surface-container-low flex flex-col max-md:border-b max-md:border-outline-variant/30 md:border-r md:border-outline-variant/30">
        <div className="p-8 pb-4 max-md:hidden">
          <h1 className="text-2xl font-black text-on-surface tracking-tight">Configuration</h1>
          <p className="text-[10px] font-mono text-primary font-bold mt-1 uppercase tracking-widest">System Node: IES-EV</p>
        </div>
        <div className="overflow-y-auto px-4 py-2 space-y-1 no-scrollbar pb-10 max-md:flex max-md:flex-row max-md:overflow-x-auto max-md:overflow-y-hidden max-md:space-y-0 max-md:gap-2 max-md:px-3 max-md:py-3 max-md:pb-3">
          {tabList.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full max-md:w-auto flex items-center gap-3 px-4 py-3 max-md:px-3 rounded-xl transition-all duration-200 text-left shrink-0 ${
                activeTab === tab.id
                  ? 'bg-primary/10 text-primary shadow-sm'
                  : 'text-on-surface-variant hover:bg-surface-variant/40 hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[20px]" style={{ fontVariationSettings: activeTab === tab.id ? "'FILL' 1" : "'FILL' 0" }}>
                {tab.icon}
              </span>
              <span className={`text-[13px] font-semibold max-md:hidden ${activeTab === tab.id ? 'font-black' : ''}`}>
                {tab.label}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area - Ultra Minimalist (No Cards) */}
      <div className="flex-1 overflow-y-auto no-scrollbar bg-surface-container-lowest flex flex-col items-center">
        <div className="w-full max-w-2xl px-10 py-12 my-auto">
          
          {/* Header */}
          <div className="mb-16">
            <h2 className="text-4xl font-black text-on-surface tracking-tighter mb-4">
              {tabList.find(t => t.id === activeTab)?.label}
            </h2>
            <div className="h-1 w-12 bg-primary mb-6" />
            <p className="text-sm text-on-surface-variant leading-relaxed opacity-80">
              Fine-tune the intelligent core of your EV experience. These parameters directly influence model inference, 
              simulation fidelity, and hardware-software interaction layers.
            </p>
          </div>

          <div className="flex flex-col gap-16">

            {/* ==== GENERAL SETTINGS ==== */}
            {activeTab === 'General' && (
              <>
                <Section title="Localization">
                  <SelectRow 
                    label="System Language" 
                    value={settings.language} 
                    onChange={(v) => updateSetting('language', v)}
                    options={['English (US)', 'Dutch (NL)', 'German (DE)']}
                  />
                  <SelectRow 
                    label="Measurement Standards" 
                    value={settings.units} 
                    onChange={(v) => updateSetting('units', v)}
                    options={['Metric (km, °C)', 'Imperial (mi, °F)']}
                  />
                </Section>

                <Section title="User Experience">
                  <ToggleRow 
                    label="System Notifications" 
                    description="Real-time telemetry alerts and threshold warnings." 
                    value={settings.notifications} 
                    onChange={(v) => updateSetting('notifications', v)} 
                  />
                  <ToggleRow 
                    label="Compact UI Density" 
                    description="Maximize information display for professional monitoring." 
                    value={settings.compactDensity} 
                    onChange={(v) => updateSetting('compactDensity', v)} 
                  />
                </Section>
              </>
            )}

            {/* ==== VEHICLE SETTINGS ==== */}
            {activeTab === 'Vehicle' && (
              <>
                <Section title="Telemetry Core">
                  <div className="flex flex-col gap-4">
                    <p className="text-[14px] font-bold text-on-surface">Prediction Fidelity Mode</p>
                    <div className="grid grid-cols-3 gap-2">
                       {['Fast', 'Balanced', 'Accurate'].map((m) => (
                         <button
                           key={m}
                           onClick={() => updateSetting('predictionMode', m)}
                           className={`py-3 px-4 rounded-lg border text-sm font-black transition-all ${
                             settings.predictionMode === m 
                             ? 'bg-primary border-primary text-on-primary shadow-lg shadow-primary/20 scale-[1.02]' 
                             : 'bg-transparent border-outline-variant/30 text-on-surface-variant hover:border-primary/40'
                           }`}
                         >
                           {m.toUpperCase()}
                         </button>
                       ))}
                    </div>
                    <p className="text-[11px] font-mono text-on-surface-variant/60 uppercase tracking-widest mt-1">
                      Higher fidelity increases CPU load and model inference latency.
                    </p>
                  </div>

                  <div className="h-px bg-outline-variant/20 my-2" />

                  <ToggleRow 
                    label="Simulated Hardware Acceleration" 
                    description="Emulate RTOS timing for more precise energy draw calculations." 
                    value={true} 
                    onChange={() => {}} 
                  />
                </Section>
              </>
            )}

            {/* ==== NAVIGATION SETTINGS ==== */}
            {activeTab === 'Navigation' && (
              <>
                <Section title="Routing Logic">
                  <ToggleRow 
                    label="Avoid Toll Infrastructure" 
                    description="Prioritize public transit arteries and local highways." 
                    value={settings.avoidTolls} 
                    onChange={(v) => updateSetting('avoidTolls', v)} 
                  />
                  <ToggleRow 
                    label="Advanced Terrain Analysis" 
                    description="Calculate grade-adjusted energy cost for every routing segment." 
                    value={settings.terrainAware} 
                    onChange={(v) => updateSetting('terrainAware', v)} 
                  />
                </Section>

                <Section title="Charging Proximity">
                  <ToggleRow 
                    label="Prioritize DC Fast Chargers" 
                    description="Only show Level 3 stations capable of >50kW delivery." 
                    value={settings.fastChargingOnly} 
                    onChange={(v) => updateSetting('fastChargingOnly', v)} 
                  />
                </Section>
              </>
            )}

            {/* ==== AI & ML SETTINGS ==== */}
            {activeTab === 'AI' && (
              <>
                <Section title="Model Calibration">
                  <div className="flex flex-col gap-8">
                    <div className="space-y-4">
                      <div className="flex justify-between items-end">
                        <p className="text-[14px] font-bold text-on-surface uppercase tracking-tight">Confidence Threshold</p>
                        <span className="text-3xl font-black text-primary font-mono tabular-nums">{settings.confidenceThreshold}%</span>
                      </div>
                      <input
                        type="range"
                        min="50"
                        max="99"
                        value={settings.confidenceThreshold}
                        onChange={(e) => updateSetting('confidenceThreshold', parseInt(e.target.value))}
                        className="w-full h-1.5 bg-outline-variant/20 rounded-full appearance-none cursor-pointer accent-primary"
                      />
                      <p className="text-[11px] font-mono text-on-surface-variant/60 leading-relaxed uppercase tracking-widest">
                        Values above 90% rely heavily on physics fallback for conservative estimates.
                      </p>
                    </div>
                  </div>
                </Section>
              </>
            )}

            {/* ==== CHARGING SETTINGS ==== */}
            {activeTab === 'Charging' && (
              <>
                <Section title="Hardware Constraints">
                   <div className="space-y-10">
                      <SliderRow 
                        label="Target SoC Limit" 
                        value={settings.targetSoC} 
                        onChange={(v) => updateSetting('targetSoC', v)} 
                        min={60} max={100} 
                        unit="%"
                      />
                      <SliderRow 
                        label="Critical Safety Reserve" 
                        value={settings.minBatteryReserve} 
                        onChange={(v) => updateSetting('minBatteryReserve', v)} 
                        min={5} max={30} 
                        unit="%"
                        danger
                      />
                   </div>
                </Section>
              </>
            )}

            {/* ==== ACCOUNT SETTINGS ==== */}
            {activeTab === 'Account' && (
              <>
                <Section title="Data Continuity">
                  <ToggleRow 
                    label="Vector Database Sync" 
                    description="Persist vehicle learn-patterns and routing history to cloud nodes." 
                    value={settings.dataSync} 
                    onChange={(v) => updateSetting('dataSync', v)} 
                  />
                </Section>

                <Section title="Security Operations">
                   <div className="flex flex-col gap-6">
                      <div className="p-6 bg-error/5 border border-error/20 rounded-lg">
                        <h4 className="text-xs font-black text-error uppercase tracking-widest mb-2">Destructive Actions</h4>
                        <p className="text-xs text-on-surface-variant mb-6 opacity-70">Resetting will wipe all local ML training weights and system configuration.</p>
                        <div className="flex gap-3">
                          <button className="px-5 py-2.5 bg-error text-white text-[11px] font-black uppercase tracking-widest rounded hover:brightness-110 transition-all">
                            Purge All Data
                          </button>
                          <button className="px-5 py-2.5 border border-error text-error text-[11px] font-black uppercase tracking-widest rounded hover:bg-error/10 transition-all">
                            De-authorize Node
                          </button>
                        </div>
                      </div>
                   </div>
                </Section>
              </>
            )}

            {/* ==== ADVANCED SETTINGS ==== */}
            {activeTab === 'Advanced' && (
              <>
                <Section title="Debug Middleware">
                  <div className="p-6 bg-[#0a0a0a] border border-primary/20 rounded-lg relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent pointer-events-none" />
                    <div className="relative z-10 space-y-6">
                      <ToggleRow 
                        label="Raw Distribution Overlay" 
                        description="Expose neural network probability histograms for energy inference." 
                        value={settings.livePredictions} 
                        onChange={(v) => updateSetting('livePredictions', v)} 
                        minimal 
                      />
                      <div className="h-px bg-white/5" />
                      <ToggleRow 
                        label="Dependency Log Trace" 
                        description="Verbose logging of external API handshakes and response times." 
                        value={settings.logApiResp} 
                        onChange={(v) => updateSetting('logApiResp', v)} 
                        minimal
                      />
                    </div>
                  </div>
                  <div className="mt-4 flex justify-between text-[10px] font-mono text-on-surface-variant/40 uppercase tracking-widest">
                    <span>Build: 4.9.2-STABLE</span>
                    <span>Kernel: 0.12.5-MERIDIAN</span>
                  </div>
                </Section>
              </>
            )}

          </div>
        </div>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-center gap-4">
        <h3 className="text-[10px] font-mono font-black text-primary uppercase tracking-[0.25em] shrink-0">{title}</h3>
        <div className="h-px flex-1 bg-outline-variant/30" />
      </div>
      <div className="flex flex-col gap-8 px-1">
        {children}
      </div>
    </div>
  )
}

function ToggleRow({ label, description, value, onChange, minimal = false }: { label: string, description: string, value: boolean, onChange: (v: boolean) => void, minimal?: boolean }) {
  return (
    <div className="flex justify-between items-center group">
      <div className="flex-1 pr-10">
        <p className={`text-[15px] font-bold ${minimal ? 'text-white' : 'text-on-surface'}`}>{label}</p>
        <p className={`text-xs mt-0.5 leading-relaxed ${minimal ? 'text-white/40 font-mono' : 'text-on-surface-variant opacity-70'}`}>{description}</p>
      </div>
      <button
        onClick={() => onChange(!value)}
        className={`relative inline-flex h-5 w-10 shrink-0 cursor-pointer rounded-full transition-colors duration-200 ease-in-out ${
          value ? 'bg-primary' : 'bg-outline-variant/30'
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-0 transition duration-200 ease-in-out mt-0.5 ${
            value ? 'translate-x-[22px]' : 'translate-x-[2px]'
          }`}
        />
      </button>
    </div>
  )
}

function SelectRow({ label, value, onChange, options }: { label: string, value: string, onChange: (v: string) => void, options: string[] }) {
  return (
    <div className="flex justify-between items-center">
      <p className="text-[14px] font-bold text-on-surface">{label}</p>
      <select 
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-transparent border-b border-outline-variant/50 text-sm font-black text-primary py-1 px-2 outline-none focus:border-primary cursor-pointer hover:border-primary/70 transition-colors"
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )
}

function SliderRow({ label, value, onChange, min, max, unit, danger = false }: { label: string, value: number, onChange: (v: number) => void, min: number, max: number, unit: string, danger?: boolean }) {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-end">
        <p className="text-[14px] font-bold text-on-surface uppercase tracking-tight">{label}</p>
        <span className={`text-2xl font-black font-mono tabular-nums ${danger ? 'text-error' : 'text-on-surface'}`}>{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className={`w-full h-1 bg-outline-variant/20 rounded-full appearance-none cursor-pointer ${danger ? 'accent-error' : 'accent-primary'}`}
      />
    </div>
  )
}

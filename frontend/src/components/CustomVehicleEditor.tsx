import { useVehicle } from '../contexts/VehicleContext';
import { useState } from 'react';
import useWindowSize from '../hooks/useWindowSize';

/*
DEMO SCRIPT — CUSTOM LAB VEHICLE:

1. Select "Custom Lab" from dropdown
2. Set SoH slider to 60%
   → Range drops to ~180km
   → Warning banner appears: "⚠ Degraded battery"

3. Go to Route Planner → Predict a 50km trip
   → Energy: ~20+ kWh (vs 12.5kWh for Model V)

4. Go back to Custom Lab, increase mass to 3500kg
   → Range drops further, power draw increases

5. Predict same 50km trip again → energy goes even higher

6. Reset to defaults → everything snaps back instantly
*/

/* ─── Slider ─── */
function LabSlider({ label, value, min, max, step, unit, onChange, markers }: any) {
  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex justify-between items-center">
        <span className="text-[11px] text-slate-400 uppercase tracking-wider">{label}</span>
        <span className="text-xs font-mono font-bold text-white bg-white/8 px-2.5 py-1 rounded-lg">
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#A855F7] [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-lg"
        style={{
          background: `linear-gradient(to right, #A855F7 0%, #A855F7 ${((value - min) / (max - min)) * 100}%, #31353e ${((value - min) / (max - min)) * 100}%, #31353e 100%)`
        }}
      />
      {markers && (
        <div className="flex justify-between">
          {markers.map((m: any) => (
            <span key={m.value} className="text-[9px] text-slate-600 font-mono">{m.label}</span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Collapsible Section ─── */
function Section({ title, defaultOpen = false, children }: any) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-white/5">
      <button
        className="flex w-full justify-between items-center px-5 py-4 outline-none"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="text-[11px] font-bold text-white uppercase tracking-widest">{title}</span>
        <span className="text-slate-500 text-base leading-none w-5 h-5 flex items-center justify-center rounded bg-white/5">
          {isOpen ? '−' : '+'}
        </span>
      </button>
      {isOpen && (
        <div className="px-5 pb-5 flex flex-col gap-5">
          {children}
        </div>
      )}
    </div>
  );
}

/* ─── Blueprint SVG Wireframe ─── */
function BlueprintWireframe({ soh, regenActive }: { soh: number; regenActive: boolean }) {
  const sohColor = soh >= 80 ? '#A855F7' : soh >= 65 ? '#FFB800' : '#FF4444';

  return (
    <div
      className="relative mx-4 rounded-xl overflow-hidden border border-[#A855F7]/20"
      style={{ height: 160, background: 'linear-gradient(135deg, rgba(168,85,247,0.04) 0%, rgba(10,12,18,0.95) 100%)' }}
    >
      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-[#A855F7]/60 rounded-tl" />
      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-[#A855F7]/60 rounded-tr" />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-[#A855F7]/60 rounded-bl" />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-[#A855F7]/60 rounded-br" />

      <div className="absolute top-2 left-3 flex items-center gap-1.5 z-10">
        <div className="w-1.5 h-1.5 rounded-full bg-[#A855F7] animate-pulse" />
        <span className="text-[8px] font-mono font-bold text-[#A855F7] uppercase tracking-widest">Research Lab — Model R</span>
      </div>

      <svg viewBox="0 0 360 120" className="absolute inset-0 w-full h-full" style={{ padding: '24px 16px 28px' }}>
        <path
          d="M50,90 L50,68 L88,42 L180,34 L272,42 L310,68 L310,90 Z"
          fill="rgba(168,85,247,0.05)"
          stroke={sohColor}
          strokeWidth="1.5"
          strokeDasharray="7 4"
          style={{ filter: `drop-shadow(0 0 4px ${sohColor}60)` }}
        />
        <ellipse cx="105" cy="93" rx="22" ry="13" fill="none" stroke={sohColor} strokeWidth="1.2" opacity="0.8" />
        <ellipse cx="255" cy="93" rx="22" ry="13" fill="none" stroke={sohColor} strokeWidth="1.2" opacity="0.8" />
        <ellipse cx="105" cy="93" rx="12" ry="7" fill="none" stroke={sohColor} strokeWidth="0.7" opacity="0.4" strokeDasharray="3 3" />
        <ellipse cx="255" cy="93" rx="12" ry="7" fill="none" stroke={sohColor} strokeWidth="0.7" opacity="0.4" strokeDasharray="3 3" />
        <rect x="125" y="84" width="110" height="6" rx="2" fill={`${sohColor}18`} stroke={sohColor} strokeWidth="0.8" opacity="0.6" />
        <text x="180" y="72" textAnchor="middle" fontFamily="monospace" fontSize="8" fill={sohColor} opacity="0.75" letterSpacing="3">
          CONFIGURING...
        </text>
      </svg>

      <div className="absolute bottom-2 left-3 flex items-center gap-1 px-1.5 py-0.5 rounded" style={{ background: `${sohColor}18`, border: `1px solid ${sohColor}40` }}>
        <span className="text-[8px] font-mono font-bold uppercase" style={{ color: sohColor }}>SoH {soh}%</span>
      </div>
      <div
        className="absolute bottom-2 right-3 flex items-center gap-1 px-1.5 py-0.5 rounded"
        style={{
          background: regenActive ? 'rgba(0,255,136,0.10)' : 'rgba(255,255,255,0.04)',
          border: `1px solid ${regenActive ? 'rgba(0,255,136,0.30)' : 'rgba(255,255,255,0.10)'}`,
        }}
      >
        <div className={`w-1.5 h-1.5 rounded-full ${regenActive ? 'bg-green-400 animate-pulse' : 'bg-slate-600'}`} />
        <span className={`text-[8px] font-mono font-bold uppercase ${regenActive ? 'text-green-400' : 'text-slate-500'}`}>
          Regen {regenActive ? 'ON' : 'OFF'}
        </span>
      </div>
    </div>
  );
}

/* ─── Main Component ─── */
export default function CustomVehicleEditor() {
  const { vehicle, isCustomMode, updateCustomVehicle, isLabMinimized, setLabMinimized } = useVehicle();
  const { isMobile } = useWindowSize();
  const [saving, setSaving] = useState(false);

  if (!isCustomMode || isLabMinimized) return null;

  const resetToDefaults = () => {
    updateCustomVehicle({
      battery: { capacity_kwh: 75, soc_percent: 76, soh_percent: 85, temperature_c: 30 },
      specs: { mass_kg: 2000, drag_coefficient: 0.35, motor_efficiency: 0.88, regen_efficiency: 0.70 },
      realtime: { power_draw_kw: 8.0, regen_active: true, cabin_hvac_kw: 1.8 }
    });
  };

  const saveProfile = () => {
    const snapshot = {
      battery: { ...vehicle.battery },
      specs: { ...vehicle.specs },
      realtime: {
        power_draw_kw: vehicle.realtime.power_draw_kw,
        regen_active: vehicle.realtime.regen_active,
        cabin_hvac_kw: vehicle.realtime.cabin_hvac_kw,
      },
    };
    localStorage.setItem('custom-lab-saved', JSON.stringify(snapshot));
    setSaving(true);
    setTimeout(() => {
      setSaving(false);
      setLabMinimized(true);
    }, 1000);
  };

  const soh = vehicle.battery.soh_percent;
  const soc = vehicle.battery.soc_percent;
  const rangeDiff = vehicle.range_km - 312;
  const rangePercent = Math.round((rangeDiff / 312) * 100);
  const predictedEnergy = ((50 * 1000) / (vehicle.range_km * 1000 / vehicle.battery.capacity_kwh)).toFixed(1);

  if (isMobile) {
    return (
      <>
        {/* Scrim */}
        <div
          className="fixed inset-0 z-[149] bg-black/60"
          onClick={() => setLabMinimized(true)}
        />
        {/* Bottom sheet */}
        <div
          className="fixed left-0 right-0 bottom-0 z-[150] bg-[#181c24] rounded-t-3xl flex flex-col"
          style={{
            height: '78dvh',
            animation: 'sheetUp 220ms cubic-bezier(0.32,0.72,0,1)',
            paddingBottom: 'env(safe-area-inset-bottom)',
          }}
        >
          {/* Drag handle */}
          <div className="flex justify-center pt-3 pb-1 shrink-0">
            <div className="w-10 h-1 rounded-full bg-white/20" />
          </div>

          {/* Header */}
          <div className="flex items-center justify-between px-5 py-3 border-b border-white/5 shrink-0">
            <div>
              <h2 className="text-sm font-bold text-white tracking-tight">⚗️ Custom Lab</h2>
              <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-0.5">Live Parameter Editor</p>
            </div>
            <button
              onClick={() => setLabMinimized(true)}
              className="w-8 h-8 rounded-full bg-white/8 flex items-center justify-center text-slate-400 active:scale-95 transition-transform"
            >
              <span className="material-symbols-outlined text-[18px]">close</span>
            </button>
          </div>

          {/* Stat pills */}
          <div className="flex gap-2.5 px-4 py-3 shrink-0">
            <div className="flex-1 bg-white/5 rounded-xl px-3 py-2.5 border border-white/8 text-center">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Range</p>
              <p className="text-xs font-mono font-bold text-white">{vehicle.range_km} km</p>
            </div>
            <div
              className="flex-1 rounded-xl px-3 py-2.5 border text-center"
              style={{
                background: soc >= 60 ? 'rgba(0,255,136,0.07)' : soc >= 30 ? 'rgba(255,184,0,0.07)' : 'rgba(255,68,68,0.07)',
                borderColor: soc >= 60 ? 'rgba(0,255,136,0.22)' : soc >= 30 ? 'rgba(255,184,0,0.22)' : 'rgba(255,68,68,0.22)',
              }}
            >
              <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">SoC</p>
              <p className="text-xs font-mono font-bold" style={{ color: soc >= 60 ? '#00FF88' : soc >= 30 ? '#FFB800' : '#FF4444' }}>{soc}%</p>
            </div>
            <div className="flex-1 bg-white/5 rounded-xl px-3 py-2.5 border border-white/8 text-center">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Cap.</p>
              <p className="text-xs font-mono font-bold text-white">{vehicle.battery.capacity_kwh} kWh</p>
            </div>
          </div>

          {/* Reset / Save */}
          <div className="flex gap-3 px-4 pb-3 shrink-0">
            <button
              onClick={resetToDefaults}
              className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl border border-white/10 text-slate-300 active:bg-white/5 transition-all font-mono text-[11px] font-bold uppercase tracking-widest"
            >
              <span className="leading-none">↺</span> Reset
            </button>
            <button
              onClick={saveProfile}
              className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-mono text-[11px] font-bold uppercase tracking-widest transition-all ${
                saving
                  ? 'bg-green-500/20 border border-green-500/40 text-green-400'
                  : 'bg-[#A855F7]/20 border border-[#A855F7]/50 text-[#A855F7]'
              }`}
            >
              <span className="leading-none">{saving ? '✓' : '⬇'}</span>
              {saving ? 'Saved!' : 'Save & Close'}
            </button>
          </div>

          {/* Scrollable sliders */}
          <div className="flex-1 overflow-y-auto no-scrollbar border-t border-white/5">
            <Section title="Battery Health" defaultOpen={true}>
              <LabSlider
                label="State of Charge (SoC)" value={soc} min={0} max={100} step={1} unit="%"
                markers={[{ value: 0, label: '0%' }, { value: 20, label: 'Low' }, { value: 50, label: '50%' }, { value: 80, label: 'Good' }, { value: 100, label: 'Full' }]}
                onChange={(v: number) => updateCustomVehicle({ battery: { soc_percent: v } })}
              />
              <LabSlider
                label="State of Health (SoH)" value={soh} min={50} max={100} step={1} unit="%"
                markers={[{ value: 50, label: '50%' }, { value: 75, label: '75%' }, { value: 100, label: '100%' }]}
                onChange={(v: number) => updateCustomVehicle({ battery: { soh_percent: v } })}
              />
              <LabSlider
                label="Battery Capacity" value={vehicle.battery.capacity_kwh} min={40} max={150} step={5} unit=" kWh"
                onChange={(v: number) => updateCustomVehicle({ battery: { capacity_kwh: v } })}
              />
              <LabSlider
                label="Temperature" value={vehicle.battery.temperature_c} min={10} max={80} step={1} unit="°C"
                onChange={(v: number) => updateCustomVehicle({ battery: { temperature_c: v } })}
              />
            </Section>
            <Section title="Vehicle Mass & Aero" defaultOpen={false}>
              <LabSlider
                label="Vehicle Mass" value={vehicle.specs.mass_kg} min={1200} max={4000} step={50} unit=" kg"
                markers={[{ value: 1200, label: 'Compact' }, { value: 2500, label: 'SUV' }, { value: 4000, label: 'Heavy' }]}
                onChange={(v: number) => updateCustomVehicle({ specs: { mass_kg: v } })}
              />
              <LabSlider
                label="Drag Coefficient (Cd)" value={vehicle.specs.drag_coefficient} min={0.20} max={0.55} step={0.01} unit=""
                markers={[{ value: 0.20, label: 'Aero' }, { value: 0.35, label: 'Avg' }, { value: 0.55, label: 'Brick' }]}
                onChange={(v: number) => updateCustomVehicle({ specs: { drag_coefficient: v } })}
              />
            </Section>
            <Section title="Motor & Drivetrain" defaultOpen={false}>
              <LabSlider
                label="Motor Efficiency" value={Math.round(vehicle.specs.motor_efficiency * 100)} min={70} max={98} step={1} unit="%"
                onChange={(v: number) => updateCustomVehicle({ specs: { motor_efficiency: v / 100 } })}
              />
              <LabSlider
                label="Regen Efficiency" value={Math.round(vehicle.specs.regen_efficiency * 100)} min={40} max={80} step={1} unit="%"
                onChange={(v: number) => updateCustomVehicle({ specs: { regen_efficiency: v / 100 } })}
              />
              <div
                className="flex justify-between items-center cursor-pointer py-1"
                onClick={() => updateCustomVehicle({ realtime: { regen_active: !vehicle.realtime.regen_active } })}
              >
                <span className="text-[11px] text-slate-400 uppercase tracking-wider">Regen Brake Active</span>
                <div className={`w-9 h-5 rounded-full p-0.5 transition-colors ${vehicle.realtime.regen_active ? 'bg-[#A855F7]' : 'bg-white/10'}`}>
                  <div className={`w-4 h-4 rounded-full bg-white transition-transform shadow ${vehicle.realtime.regen_active ? 'translate-x-4' : 'translate-x-0'}`} />
                </div>
              </div>
            </Section>
            <Section title="Power & Load" defaultOpen={false}>
              <LabSlider
                label="Avg Power Draw" value={vehicle.realtime.power_draw_kw} min={1} max={50} step={0.5} unit=" kW"
                onChange={(v: number) => updateCustomVehicle({ realtime: { power_draw_kw: v } })}
              />
              <LabSlider
                label="HVAC Load" value={vehicle.realtime.cabin_hvac_kw} min={0} max={5} step={0.1} unit=" kW"
                onChange={(v: number) => updateCustomVehicle({ realtime: { cabin_hvac_kw: v } })}
              />
            </Section>
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="fixed right-0 top-0 bottom-0 w-[320px] bg-[#181c24] border-l border-white/5 z-[45] flex flex-col pt-14">

      {/* ── Header ── */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/5 shrink-0">
        <div>
          <h2 className="text-sm font-bold text-white tracking-tight">⚗️ Custom Lab</h2>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-0.5">Live Parameter Editor</p>
        </div>
      </div>

      {/* ── Blueprint Wireframe ── */}
      <div className="shrink-0 px-0 py-4">
        <BlueprintWireframe soh={soh} regenActive={vehicle.realtime.regen_active} />
      </div>

      {/* ── Stat pills ── */}
      <div className="flex gap-2.5 px-4 shrink-0">
        <div className="flex-1 bg-white/5 rounded-xl px-3 py-3 border border-white/8 text-center">
          <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Range</p>
          <p className="text-xs font-mono font-bold text-white">{vehicle.range_km} km</p>
        </div>
        <div
          className="flex-1 rounded-xl px-3 py-3 border text-center"
          style={{
            background: soc >= 60 ? 'rgba(0,255,136,0.07)' : soc >= 30 ? 'rgba(255,184,0,0.07)' : 'rgba(255,68,68,0.07)',
            borderColor: soc >= 60 ? 'rgba(0,255,136,0.22)' : soc >= 30 ? 'rgba(255,184,0,0.22)' : 'rgba(255,68,68,0.22)',
          }}
        >
          <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">SoC</p>
          <p className="text-xs font-mono font-bold" style={{ color: soc >= 60 ? '#00FF88' : soc >= 30 ? '#FFB800' : '#FF4444' }}>{soc}%</p>
        </div>
        <div className="flex-1 bg-white/5 rounded-xl px-3 py-3 border border-white/8 text-center">
          <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Cap.</p>
          <p className="text-xs font-mono font-bold text-white">{vehicle.battery.capacity_kwh} kWh</p>
        </div>
      </div>

      {/* ── Reset / Save ── */}
      <div className="flex gap-3 px-4 pt-3 pb-4 shrink-0">
        <button
          onClick={resetToDefaults}
          className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl border border-white/10 text-slate-300 hover:text-white hover:border-white/25 hover:bg-white/5 transition-all font-mono text-[11px] font-bold uppercase tracking-widest"
        >
          <span className="leading-none">↺</span> Reset
        </button>
        <button
          onClick={saveProfile}
          className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-mono text-[11px] font-bold uppercase tracking-widest transition-all ${
            saving
              ? 'bg-green-500/20 border border-green-500/40 text-green-400'
              : 'bg-[#A855F7]/20 border border-[#A855F7]/50 text-[#A855F7] hover:bg-[#A855F7]/30 hover:border-[#A855F7]/70'
          }`}
        >
          <span className="leading-none">{saving ? '✓' : '⬇'}</span>
          {saving ? 'Saving...' : 'Save & Minimize'}
        </button>
      </div>

      {/* ── Scrollable sections + impact preview ── */}
      <div className="flex-1 overflow-y-auto no-scrollbar border-t border-white/5">

        <Section title="Battery Health" defaultOpen={true}>
          <LabSlider
            label="State of Charge (SoC)" value={soc} min={0} max={100} step={1} unit="%"
            markers={[{ value: 0, label: '0%' }, { value: 20, label: 'Low' }, { value: 50, label: '50%' }, { value: 80, label: 'Good' }, { value: 100, label: 'Full' }]}
            onChange={(v: number) => updateCustomVehicle({ battery: { soc_percent: v } })}
          />
          <LabSlider
            label="State of Health (SoH)" value={soh} min={50} max={100} step={1} unit="%"
            markers={[{ value: 50, label: '50%' }, { value: 75, label: '75%' }, { value: 100, label: '100%' }]}
            onChange={(v: number) => updateCustomVehicle({ battery: { soh_percent: v } })}
          />
          <LabSlider
            label="Battery Capacity" value={vehicle.battery.capacity_kwh} min={40} max={150} step={5} unit=" kWh"
            onChange={(v: number) => updateCustomVehicle({ battery: { capacity_kwh: v } })}
          />
          <LabSlider
            label="Temperature" value={vehicle.battery.temperature_c} min={10} max={80} step={1} unit="°C"
            onChange={(v: number) => updateCustomVehicle({ battery: { temperature_c: v } })}
          />
        </Section>

        <Section title="Vehicle Mass & Aero" defaultOpen={false}>
          <LabSlider
            label="Vehicle Mass" value={vehicle.specs.mass_kg} min={1200} max={4000} step={50} unit=" kg"
            markers={[{ value: 1200, label: 'Compact' }, { value: 2500, label: 'SUV' }, { value: 4000, label: 'Heavy' }]}
            onChange={(v: number) => updateCustomVehicle({ specs: { mass_kg: v } })}
          />
          <LabSlider
            label="Drag Coefficient (Cd)" value={vehicle.specs.drag_coefficient} min={0.20} max={0.55} step={0.01} unit=""
            markers={[{ value: 0.20, label: 'Aero' }, { value: 0.35, label: 'Avg' }, { value: 0.55, label: 'Brick' }]}
            onChange={(v: number) => updateCustomVehicle({ specs: { drag_coefficient: v } })}
          />
        </Section>

        <Section title="Motor & Drivetrain" defaultOpen={false}>
          <LabSlider
            label="Motor Efficiency" value={Math.round(vehicle.specs.motor_efficiency * 100)} min={70} max={98} step={1} unit="%"
            onChange={(v: number) => updateCustomVehicle({ specs: { motor_efficiency: v / 100 } })}
          />
          <LabSlider
            label="Regen Efficiency" value={Math.round(vehicle.specs.regen_efficiency * 100)} min={40} max={80} step={1} unit="%"
            onChange={(v: number) => updateCustomVehicle({ specs: { regen_efficiency: v / 100 } })}
          />
          <div
            className="flex justify-between items-center cursor-pointer py-1"
            onClick={() => updateCustomVehicle({ realtime: { regen_active: !vehicle.realtime.regen_active } })}
          >
            <span className="text-[11px] text-slate-400 uppercase tracking-wider">Regen Brake Active</span>
            <div className={`w-9 h-5 rounded-full p-0.5 transition-colors ${vehicle.realtime.regen_active ? 'bg-[#A855F7]' : 'bg-white/10'}`}>
              <div className={`w-4 h-4 rounded-full bg-white transition-transform shadow ${vehicle.realtime.regen_active ? 'translate-x-4' : 'translate-x-0'}`} />
            </div>
          </div>
        </Section>

        <Section title="Power & Load" defaultOpen={false}>
          <LabSlider
            label="Avg Power Draw" value={vehicle.realtime.power_draw_kw} min={1} max={50} step={0.5} unit=" kW"
            onChange={(v: number) => updateCustomVehicle({ realtime: { power_draw_kw: v } })}
          />
          <LabSlider
            label="HVAC Load" value={vehicle.realtime.cabin_hvac_kw} min={0} max={5} step={0.1} unit=" kW"
            onChange={(v: number) => updateCustomVehicle({ realtime: { cabin_hvac_kw: v } })}
          />
        </Section>

        {/* ── Live Impact Preview ── */}
        <div className="mx-4 my-4 rounded-xl border border-white/8 overflow-hidden">
          <div
            className="px-4 py-3 border-b border-white/5"
            style={{ background: 'linear-gradient(135deg, rgba(168,85,247,0.10) 0%, transparent 100%)' }}
          >
            <p className="text-[10px] text-[#A855F7] uppercase tracking-widest font-bold">Live Impact Preview</p>
          </div>
          <div className="px-4 py-4 space-y-3 bg-white/3">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Est. Range</span>
              <span className="text-white font-mono font-bold">{vehicle.range_km} km</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">vs Model V baseline</span>
              <span className={`font-mono ${rangeDiff >= 0 ? 'text-green-400' : 'text-amber-400'}`}>
                {rangeDiff >= 0 ? '+' : ''}{rangeDiff} km ({rangePercent >= 0 ? '+' : ''}{rangePercent}%)
              </span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Available energy</span>
              <span className="text-white font-mono">{(vehicle.battery.capacity_kwh * (soh / 100) * (soc / 100)).toFixed(1)} kWh</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Predicted 50 km energy</span>
              <span className="text-white font-mono">~{predictedEnergy} kWh</span>
            </div>
            <div className="pt-0.5">
              <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(100, (vehicle.range_km / 400) * 100)}%`,
                    background: rangeDiff >= 0 ? '#A855F7' : '#FFB800',
                  }}
                />
              </div>
            </div>
          </div>

          {/* Warning banners */}
          {(soc < 40 || soh < 80 || vehicle.specs.mass_kg > 2500) && (
            <div className="px-4 pb-4 space-y-2 bg-white/3">
              {soc < 20 && (
                <div className="p-2.5 rounded-lg bg-red-400/10 border border-red-400/20">
                  <p className="text-[10px] text-red-400 font-mono">⚠ Critical charge — find a charger</p>
                </div>
              )}
              {soc >= 20 && soc < 40 && (
                <div className="p-2.5 rounded-lg bg-amber-400/10 border border-amber-400/20">
                  <p className="text-[10px] text-amber-400 font-mono">⚠ Low charge — limited range</p>
                </div>
              )}
              {soh < 80 && (
                <div className="p-2.5 rounded-lg bg-amber-400/10 border border-amber-400/20">
                  <p className="text-[10px] text-amber-400 font-mono">⚠ Degraded battery simulated</p>
                </div>
              )}
              {vehicle.specs.mass_kg > 2500 && (
                <div className="p-2.5 rounded-lg bg-orange-400/10 border border-orange-400/20">
                  <p className="text-[10px] text-orange-400 font-mono">ℹ Heavy vehicle — high energy use</p>
                </div>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

import { useState } from 'react';
import { useVehicle } from '../contexts/VehicleContext';

/*
DEMO SCRIPT — CUSTOM LAB VEHICLE:

1. Select "Custom Lab" from dropdown
2. Say: "Watch what happens when I configure a vehicle
        with severely degraded battery..."

3. Set SoH slider to 60%
   → Range drops to ~180km
   → Warning banner appears: "⚠ Degraded battery"
   → Cell voltage map turns mostly red

4. Go to Route Planner
   → Predict a 50km trip
   → Energy: ~20+ kWh (vs 12.5kWh for Model V)
   → Say: "That's 60% more energy for the same trip"

5. Go back to Custom Lab, increase mass to 3500kg
   → Range drops further
   → Power draw increases
   → Say: "Now combining degradation with heavy payload..."

6. Predict same 50km trip again
   → Energy goes even higher
   → Say: "Our ML system adapts to ANY combination of
           real-world factors — this is what Tesla's BMS
           cannot do in isolation."

7. Reset to defaults
   → Everything snaps back
   → "And we can reset to baseline instantly."
*/

/* ─── Slider ─── */
function LabSlider({ label, value, min, max, step, unit, onChange, markers }: any) {
  return (
    <div className="mb-5">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[11px] text-slate-400 uppercase tracking-wider">{label}</span>
        <span className="text-sm font-mono font-bold text-white bg-surface-container px-2 py-0.5 rounded-md">
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-surface-container-highest [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#A855F7] [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-lg"
        style={{
          background: `linear-gradient(to right, #A855F7 0%, #A855F7 ${((value - min) / (max - min)) * 100}%, #31353e ${((value - min) / (max - min)) * 100}%, #31353e 100%)`
        }}
      />
      {markers && (
        <div className="flex justify-between mt-1">
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
    <div className="border-b border-white/5 py-4 px-5">
      <button
        className="flex w-full justify-between items-center outline-none"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="text-xs font-bold text-white uppercase tracking-widest">{title}</span>
        <span className="text-slate-400 text-lg leading-none">{isOpen ? '−' : '+'}</span>
      </button>
      {isOpen && <div className="mt-6">{children}</div>}
    </div>
  );
}

/* ─── Blueprint SVG Wireframe ─── */
function BlueprintWireframe({ soh, mass, regenActive }: { soh: number; mass: number; regenActive: boolean }) {
  const sohColor = soh >= 80 ? '#A855F7' : soh >= 65 ? '#FFB800' : '#FF4444';
  const pulseOpacity = soh < 65 ? 0.9 : 0.7;

  return (
    <div
      className="relative mx-4 mt-4 mb-1 rounded-xl overflow-hidden border border-[#A855F7]/20"
      style={{ height: 200, background: 'linear-gradient(135deg, rgba(168,85,247,0.04) 0%, rgba(10,12,18,0.95) 100%)' }}
    >
      {/* Corner accents */}
      <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-[#A855F7]/60 rounded-tl" />
      <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-[#A855F7]/60 rounded-tr" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-[#A855F7]/60 rounded-bl" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-[#A855F7]/60 rounded-br" />

      {/* Top label */}
      <div className="absolute top-2.5 left-4 flex items-center gap-1.5 z-10">
        <div className="w-1.5 h-1.5 rounded-full bg-[#A855F7] animate-pulse" />
        <span className="text-[9px] font-mono font-bold text-[#A855F7] uppercase tracking-widest">Research Lab — Model R</span>
      </div>

      {/* Aero winglet label (top-left telemetry) */}
      <div className="absolute top-8 left-3 z-10">
        <span className="font-mono text-[8px] text-[#00D9FF] uppercase tracking-wider">Aero Winglet</span>
        <div className="w-12 h-px bg-gradient-to-r from-[#00D9FF]/60 to-transparent mt-0.5" />
      </div>

      {/* Rear Motor Array label (bottom-right telemetry) */}
      <div className="absolute bottom-10 right-3 z-10 text-right">
        <div className="w-16 h-px bg-gradient-to-l from-[#00D9FF]/60 to-transparent mb-0.5 ml-auto" />
        <span className="font-mono text-[8px] text-[#00D9FF] uppercase tracking-wider">Rear Motor Array</span>
      </div>

      {/* SVG Wireframe */}
      <svg
        viewBox="0 0 360 160"
        className="absolute inset-0 w-full h-full"
        style={{ padding: '28px 16px 32px' }}
      >
        {/* Body outline dashed */}
        <path
          d="M50,120 L50,92 L88,58 L180,48 L272,58 L310,92 L310,120 Z"
          fill="rgba(168,85,247,0.05)"
          stroke={sohColor}
          strokeWidth="1.5"
          strokeDasharray="7 4"
          style={{ filter: `drop-shadow(0 0 4px ${sohColor}60)` }}
        />

        {/* Wheel wells */}
        <ellipse cx="105" cy="123" rx="26" ry="16" fill="none" stroke={sohColor} strokeWidth="1.5" opacity="0.8" />
        <ellipse cx="255" cy="123" rx="26" ry="16" fill="none" stroke={sohColor} strokeWidth="1.5" opacity="0.8" />

        {/* Inner wheel circles */}
        <ellipse cx="105" cy="123" rx="14" ry="9" fill="none" stroke={sohColor} strokeWidth="0.8" opacity="0.4" strokeDasharray="3 3" />
        <ellipse cx="255" cy="123" rx="14" ry="9" fill="none" stroke={sohColor} strokeWidth="0.8" opacity="0.4" strokeDasharray="3 3" />

        {/* Cabin window line */}
        <path
          d="M105,90 L130,62 L230,62 L255,90"
          fill="none"
          stroke={sohColor}
          strokeWidth="1"
          opacity="0.5"
          strokeDasharray="4 3"
        />

        {/* Centre scan line (animated feel via low opacity) */}
        <line x1="50" y1="120" x2="310" y2="120" stroke={sohColor} strokeWidth="0.5" opacity="0.25" />

        {/* Battery pack indicator (underfloor) */}
        <rect x="120" y="112" width="120" height="8" rx="2"
          fill={`${sohColor}18`} stroke={sohColor} strokeWidth="0.8" opacity="0.6" />

        {/* Configuring label */}
        <text
          x="180" y="96"
          textAnchor="middle"
          fontFamily="monospace"
          fontSize="10"
          fill={sohColor}
          opacity={pulseOpacity}
          letterSpacing="3"
        >
          CONFIGURING...
        </text>
      </svg>

      {/* SoH pill */}
      <div
        className="absolute bottom-2.5 left-4 flex items-center gap-1.5 px-2 py-0.5 rounded"
        style={{ background: `${sohColor}18`, border: `1px solid ${sohColor}40` }}
      >
        <span className="text-[8px] font-mono font-bold uppercase" style={{ color: sohColor }}>SoH {soh}%</span>
      </div>

      {/* Regen pill */}
      <div
        className="absolute bottom-2.5 right-4 flex items-center gap-1.5 px-2 py-0.5 rounded"
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
  const { vehicle, isCustomMode, updateCustomVehicle } = useVehicle();
  const [saved, setSaved] = useState(false);

  if (!isCustomMode) return null;

  const resetToDefaults = () => {
    updateCustomVehicle({
      battery: { capacity_kwh: 75, soh_percent: 85, temperature_c: 30 },
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
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const rangeDiff = vehicle.range_km - 312;
  const rangePercent = Math.round((rangeDiff / 312) * 100);
  const predictedEnergy = ((50 * 1000) / (vehicle.range_km * 1000 / vehicle.battery.capacity_kwh)).toFixed(1);
  const soh = vehicle.battery.soh_percent;

  return (
    <div
      className="fixed right-0 top-0 bottom-0 w-[320px] bg-[#181c24] border-l border-white/5 z-[45] overflow-y-auto no-scrollbar pt-14"
      style={{
        transform: isCustomMode ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 300ms cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      {/* ── Header ── */}
      <div className="flex items-center justify-between p-5 border-b border-white/5">
        <div>
          <h2 className="text-sm font-bold text-white tracking-tight">⚗️ Custom Lab</h2>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-0.5">Live Parameter Editor</p>
        </div>
      </div>

      {/* ── Blueprint Wireframe ── */}
      <BlueprintWireframe
        soh={soh}
        mass={vehicle.specs.mass_kg}
        regenActive={vehicle.realtime.regen_active}
      />

      {/* ── Live stat pills ── */}
      <div className="flex gap-2 px-4 mb-1 mt-2">
        <div className="flex-1 bg-surface-container rounded-lg px-3 py-2 border border-white/5 text-center">
          <p className="text-[9px] text-slate-500 uppercase tracking-wider">Range</p>
          <p className="text-xs font-mono font-bold text-white">{vehicle.range_km} km</p>
        </div>
        <div className="flex-1 bg-surface-container rounded-lg px-3 py-2 border border-white/5 text-center">
          <p className="text-[9px] text-slate-500 uppercase tracking-wider">Capacity</p>
          <p className="text-xs font-mono font-bold text-white">{vehicle.battery.capacity_kwh} kWh</p>
        </div>
        <div className="flex-1 bg-surface-container rounded-lg px-3 py-2 border border-white/5 text-center">
          <p className="text-[9px] text-slate-500 uppercase tracking-wider">Mass</p>
          <p className="text-xs font-mono font-bold text-white">{vehicle.specs.mass_kg} kg</p>
        </div>
      </div>

      {/* ── Reset / Save action bar ── */}
      <div className="flex gap-3 px-4 pt-3 pb-1">
        <button
          onClick={resetToDefaults}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl border border-white/10 text-slate-300 hover:text-white hover:border-white/25 hover:bg-white/5 transition-all font-mono text-xs font-bold uppercase tracking-widest"
        >
          <span className="text-base leading-none">↺</span> Reset
        </button>
        <button
          onClick={saveProfile}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl font-mono text-xs font-bold uppercase tracking-widest transition-all ${
            saved
              ? 'bg-green-500/20 border border-green-500/40 text-green-400'
              : 'bg-[#A855F7]/20 border border-[#A855F7]/50 text-[#A855F7] hover:bg-[#A855F7]/30 hover:border-[#A855F7]/70'
          }`}
        >
          <span className="text-base leading-none">{saved ? '✓' : '⬇'}</span>
          {saved ? 'Saved!' : 'Save'}
        </button>
      </div>

      {/* ── Parameter Sections ── */}
      <Section title="Battery Health" defaultOpen={true}>
        <LabSlider
          label="State of Health (SoH)" value={soh} min={50} max={100} step={1} unit="%"
          markers={[{ value: 50, label: '50%' }, { value: 100, label: '100%' }]}
          onChange={(v: number) => updateCustomVehicle({ battery: { soh_percent: v } })}
        />
        <LabSlider
          label="Battery Capacity" value={vehicle.battery.capacity_kwh} min={40} max={150} step={5} unit="kWh"
          onChange={(v: number) => updateCustomVehicle({ battery: { capacity_kwh: v } })}
        />
        <LabSlider
          label="Battery Temperature" value={vehicle.battery.temperature_c} min={10} max={80} step={1} unit="°C"
          onChange={(v: number) => updateCustomVehicle({ battery: { temperature_c: v } })}
        />
      </Section>

      <Section title="Vehicle Mass & Aero" defaultOpen={false}>
        <LabSlider
          label="Vehicle Mass" value={vehicle.specs.mass_kg} min={1200} max={4000} step={50} unit="kg"
          markers={[{ value: 1200, label: 'Compact' }, { value: 1600, label: 'Sedan' }, { value: 2500, label: 'SUV' }, { value: 3500, label: 'Truck' }, { value: 4000, label: 'Heavy' }]}
          onChange={(v: number) => updateCustomVehicle({ specs: { mass_kg: v } })}
        />
        <LabSlider
          label="Drag Coefficient (Cd)" value={vehicle.specs.drag_coefficient} min={0.20} max={0.55} step={0.01} unit=""
          markers={[{ value: 0.20, label: 'Aero' }, { value: 0.23, label: 'Tesla' }, { value: 0.30, label: 'Sedan' }, { value: 0.45, label: 'Truck' }, { value: 0.55, label: 'Brick' }]}
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
          className="flex justify-between items-center mb-2 mt-4 cursor-pointer"
          onClick={() => updateCustomVehicle({ realtime: { regen_active: !vehicle.realtime.regen_active } })}
        >
          <span className="text-[11px] text-slate-400 uppercase tracking-wider">Regen Brake Active</span>
          <div className={`w-8 h-4 rounded-full p-0.5 transition-colors ${vehicle.realtime.regen_active ? 'bg-[#A855F7]' : 'bg-surface-container-highest'}`}>
            <div className={`w-3 h-3 rounded-full bg-white transition-transform ${vehicle.realtime.regen_active ? 'translate-x-4' : 'translate-x-0'}`} />
          </div>
        </div>
      </Section>

      <Section title="Power & Load" defaultOpen={false}>
        <LabSlider
          label="Avg Power Draw" value={vehicle.realtime.power_draw_kw} min={1} max={50} step={0.5} unit="kW"
          onChange={(v: number) => updateCustomVehicle({ realtime: { power_draw_kw: v } })}
        />
        <LabSlider
          label="HVAC Load" value={vehicle.realtime.cabin_hvac_kw} min={0} max={5} step={0.1} unit="kW"
          onChange={(v: number) => updateCustomVehicle({ realtime: { cabin_hvac_kw: v } })}
        />
      </Section>

      {/* ── Live Impact Preview ── */}
      <div className="mt-4 mx-4 mb-8 rounded-xl border border-white/5 overflow-hidden">
        <div
          className="px-4 pt-3 pb-2 border-b border-white/5"
          style={{ background: 'linear-gradient(135deg, rgba(168,85,247,0.08) 0%, transparent 100%)' }}
        >
          <p className="text-[10px] text-[#A855F7] uppercase tracking-widest font-bold">Live Impact Preview</p>
        </div>

        <div className="p-4 space-y-2.5 bg-surface-container-lowest">
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
            <span className="text-slate-400">Predicted 50km energy</span>
            <span className="text-white font-mono">~{predictedEnergy} kWh</span>
          </div>

          {/* Range bar */}
          <div className="pt-1">
            <div className="w-full h-1 rounded-full bg-white/10 overflow-hidden">
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

        {soh < 80 && (
          <div className="mx-4 mb-3 p-2 rounded-lg bg-amber-400/10 border border-amber-400/20">
            <p className="text-[10px] text-amber-400 font-mono">⚠ Degraded battery simulated</p>
          </div>
        )}
        {vehicle.specs.mass_kg > 2500 && (
          <div className="mx-4 mb-3 p-2 rounded-lg bg-orange-400/10 border border-orange-400/20">
            <p className="text-[10px] text-orange-400 font-mono">ℹ Heavy vehicle — high energy use</p>
          </div>
        )}
      </div>
    </div>
  );
}

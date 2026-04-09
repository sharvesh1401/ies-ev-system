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

function LabSlider({ label, value, min, max, step, unit, onChange, markers }: any) {
  return (
    <div className="mb-5">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[11px] text-slate-400 uppercase tracking-wider">
          {label}
        </span>
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
            <span key={m.value} className="text-[9px] text-slate-600 font-mono">
              {m.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

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

export default function CustomVehicleEditor() {
  const { vehicle, isCustomMode, updateCustomVehicle } = useVehicle();

  if (!isCustomMode) return null;

  const resetToDefaults = () => {
    updateCustomVehicle({
      battery: { capacity_kwh: 75, soh_percent: 85, temperature_c: 30 },
      specs: { mass_kg: 2000, drag_coefficient: 0.35, motor_efficiency: 0.88, regen_efficiency: 0.70 },
      realtime: { power_draw_kw: 8.0, regen_active: true, cabin_hvac_kw: 1.8 }
    });
  };

  // calculate difference against baseline model-v-performance
  // default model v has 312km range
  const rangeDiff = vehicle.range_km - 312;
  const rangePercent = Math.round((rangeDiff / 312) * 100);
  const predictedEnergy = ((50 * 1000) / (vehicle.range_km * 1000 / vehicle.battery.capacity_kwh)).toFixed(1);

  return (
    <div 
      className="fixed right-0 top-0 bottom-0 w-[320px] bg-[#181c24] border-l border-white/5 z-[45] overflow-y-auto no-scrollbar pt-14"
      style={{
        transform: isCustomMode ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 300ms cubic-bezier(0.4, 0, 0.2, 1)'
      }}
    >
      <div className="flex items-center justify-between p-5 border-b border-white/5">
        <div>
          <h2 className="text-sm font-bold text-white tracking-tight">
            ⚗️ Custom Lab
          </h2>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-0.5">
            Live Parameter Editor
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={resetToDefaults}
            className="text-[10px] px-3 py-1.5 rounded-lg border border-outline-variant/30 text-slate-400 hover:text-white hover:border-white/20 transition-all font-mono"
          >
            Reset
          </button>
        </div>
      </div>

      <Section title="Battery Health" defaultOpen={true}>
        <LabSlider 
          label="State of Health (SoH)" value={vehicle.battery.soh_percent} min={50} max={100} step={1} unit="%"
          markers={[{value: 50, label: '50%'}, {value: 100, label: '100%'}]}
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
          markers={[{value: 1200, label: 'Compact'}, {value: 1600, label: 'Sedan'}, {value: 2500, label: 'SUV'}, {value: 3500, label: 'Truck'}, {value: 4000, label: 'Heavy'}]}
          onChange={(v: number) => updateCustomVehicle({ specs: { mass_kg: v } })}
        />
        <LabSlider 
          label="Drag Coefficient (Cd)" value={vehicle.specs.drag_coefficient} min={0.20} max={0.55} step={0.01} unit=""
          markers={[{value: 0.20, label: 'Aero'}, {value: 0.23, label: 'Tesla'}, {value: 0.30, label: 'Sedan'}, {value: 0.45, label: 'Truck'}, {value: 0.55, label: 'Brick'}]}
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
        <div className="flex justify-between items-center mb-2 mt-4 cursor-pointer" onClick={() => updateCustomVehicle({ realtime: { regen_active: !vehicle.realtime.regen_active } })}>
          <span className="text-[11px] text-slate-400 uppercase tracking-wider">
            Regen Brake Active
          </span>
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

      <div className="mt-6 p-4 bg-surface-container-lowest rounded-xl mx-4 mb-8 border border-white/5">
        <p className="text-[10px] text-slate-400 uppercase tracking-widest mb-3">
          Live Impact Preview
        </p>
        
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Est. Range</span>
            <span className="text-white font-mono font-bold">
              {vehicle.range_km} km
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">vs Model V baseline</span>
            <span className={rangeDiff >= 0 ? 'text-green-400 font-mono' : 'text-amber-400 font-mono'}>
              {rangeDiff >= 0 ? '+' : ''}{rangeDiff} km ({rangePercent >= 0 ? '+' : ''}{rangePercent}%)
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Predicted 50km energy</span>
            <span className="text-white font-mono">
              ~{predictedEnergy} kWh
            </span>
          </div>
        </div>

        {vehicle.battery.soh_percent < 80 && (
          <div className="mt-3 p-2 rounded-lg bg-amber-400/10 border border-amber-400/20">
            <p className="text-[10px] text-amber-400 font-mono">
              ⚠ Degraded battery simulated
            </p>
          </div>
        )}
        {vehicle.specs.mass_kg > 2500 && (
          <div className="mt-2 p-2 rounded-lg bg-orange-400/10 border border-orange-400/20">
            <p className="text-[10px] text-orange-400 font-mono">
              ℹ Heavy vehicle — high energy use
            </p>
          </div>
        )}
      </div>

    </div>
  );
}

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Define TS types
export interface BatteryState {
  capacity_kwh: number;
  soc_percent: number;
  soh_percent: number;
  temperature_c: number;
  voltage_v: number;
  charge_cycles: number;
  cell_deviation_mv: number;
}

export interface RealtimeData {
  power_draw_kw: number;
  regen_active: boolean;
  motor_temp_c: number;
  inverter_temp_c: number;
  coolant_temp_c: number;
  cabin_hvac_kw: number;
  auxiliary_kw: number;
  efficiency_wh_per_km: number;
  optimal_wh_per_km: number;
}

export interface VehicleSpecs {
  mass_kg: number;
  drag_coefficient: number;
  rolling_resistance: number;
  frontal_area_m2: number;
  motor_efficiency: number;
  regen_efficiency: number;
  max_power_kw: number;
  max_torque_nm: number;
}

export interface HealthData {
  soh_percent: number;
  charge_cycles: number;
  lifetime_years_remaining: number;
  degradation_rate_per_month: number;
  next_service_days: number;
  cell_voltage_map: 'balanced' | 'deviation' | 'critical';
}

export interface VehicleProfile {
  id: string;
  name: string;
  subtitle: string;
  icon: string;
  color: string;
  badge: string;
  badgeColor: string;
  battery: BatteryState;
  realtime: RealtimeData;
  specs: VehicleSpecs;
  health: HealthData;
  range_km: number;
  carImage: string;
  modelPath: string;
  description: string;
}

// Vehicle profiles data
const VEHICLE_PROFILES: Record<string, VehicleProfile> = {
  'model-v-performance': {
    id: 'model-v-performance',
    name: 'Model V - Performance',
    subtitle: 'Healthy Battery • Sport Sedan',
    icon: '⚡',
    color: '#00D9FF',
    badge: 'OPTIMAL',
    badgeColor: '#00FF88',

    battery: {
      capacity_kwh: 75,
      soc_percent: 76,
      soh_percent: 94,
      temperature_c: 29,
      voltage_v: 398,
      charge_cycles: 187,
      cell_deviation_mv: 2,
    },

    realtime: {
      power_draw_kw: 4.3,
      regen_active: true,
      motor_temp_c: 42,
      inverter_temp_c: 38,
      coolant_temp_c: 31,
      cabin_hvac_kw: 1.2,
      auxiliary_kw: 0.4,
      efficiency_wh_per_km: 142,
      optimal_wh_per_km: 118,
    },

    specs: {
      mass_kg: 1600,
      drag_coefficient: 0.28,
      rolling_resistance: 0.010,
      frontal_area_m2: 2.5,
      motor_efficiency: 0.92,
      regen_efficiency: 0.75,
      max_power_kw: 350,
      max_torque_nm: 510,
    },

    health: {
      soh_percent: 94,
      charge_cycles: 187,
      lifetime_years_remaining: 6.1,
      degradation_rate_per_month: 0.048,
      next_service_days: 124,
      cell_voltage_map: 'balanced',
    },

    range_km: 312,
    carImage: '/assets/model-v-yellow.png',
    modelPath: '/models/car.glb',
    description: 'High-performance electric sedan with healthy battery pack. Optimal efficiency and range.',
  },

  'model-s-commuter': {
    id: 'model-s-commuter',
    name: 'Model S - Commuter',
    subtitle: 'Degraded Battery • Daily Driver',
    icon: '🚙',
    color: '#FFB800',
    badge: 'DEGRADED',
    badgeColor: '#FFB800',

    battery: {
      capacity_kwh: 75,
      soc_percent: 76,
      soh_percent: 78,
      temperature_c: 38,
      voltage_v: 381,
      charge_cycles: 612,
      cell_deviation_mv: 14,
    },

    realtime: {
      power_draw_kw: 6.8,
      regen_active: true,
      motor_temp_c: 58,
      inverter_temp_c: 52,
      coolant_temp_c: 44,
      cabin_hvac_kw: 1.8,
      auxiliary_kw: 0.6,
      efficiency_wh_per_km: 198,
      optimal_wh_per_km: 118,
    },

    specs: {
      mass_kg: 1700,
      drag_coefficient: 0.32,
      rolling_resistance: 0.011,
      frontal_area_m2: 2.6,
      motor_efficiency: 0.84,
      regen_efficiency: 0.65,
      max_power_kw: 220,
      max_torque_nm: 340,
    },

    health: {
      soh_percent: 78,
      charge_cycles: 612,
      lifetime_years_remaining: 2.3,
      degradation_rate_per_month: 0.19,
      next_service_days: 12,
      cell_voltage_map: 'critical',
    },

    range_km: 220,
    carImage: '/assets/model-s-blue.png',
    modelPath: '/models/commuter.glb',
    description: 'Standard sedan with aging battery. Reduced range due to degradation.',
  },

  'model-t-cargo': {
    id: 'model-t-cargo',
    name: 'Model T - Cargo',
    subtitle: 'Heavy Duty • Electric Truck',
    icon: '🚚',
    color: '#FF6B00',
    badge: 'HEAVY',
    badgeColor: '#FF6B00',

    battery: {
      capacity_kwh: 100,
      soc_percent: 76,
      soh_percent: 90,
      temperature_c: 34,
      voltage_v: 820,
      charge_cycles: 298,
      cell_deviation_mv: 4,
    },

    realtime: {
      power_draw_kw: 18.4,
      regen_active: false,
      motor_temp_c: 78,
      inverter_temp_c: 71,
      coolant_temp_c: 58,
      cabin_hvac_kw: 3.2,
      auxiliary_kw: 2.1,
      efficiency_wh_per_km: 312,
      optimal_wh_per_km: 118,
    },

    specs: {
      mass_kg: 2800,
      drag_coefficient: 0.45,
      rolling_resistance: 0.015,
      frontal_area_m2: 3.5,
      motor_efficiency: 0.88,
      regen_efficiency: 0.52,
      max_power_kw: 560,
      max_torque_nm: 1100,
    },

    health: {
      soh_percent: 90,
      charge_cycles: 298,
      lifetime_years_remaining: 4.8,
      degradation_rate_per_month: 0.071,
      next_service_days: 67,
      cell_voltage_map: 'deviation',
    },

    range_km: 280,
    carImage: '/assets/model-t-truck.png',
    modelPath: '/models/cargo.glb',
    description: 'Heavy-duty electric truck. High energy consumption.',
  },
};

interface VehicleContextType {
  vehicle: VehicleProfile;
  currentVehicle: string;
  switchVehicle: (id: string) => void;
  allVehicles: Record<string, VehicleProfile>;
}

const VehicleContext = createContext<VehicleContextType | undefined>(undefined);

export function VehicleProvider({ children }: { children: ReactNode }) {
  const [currentVehicle, setCurrentVehicle] = useState('model-v-performance');
  const [liveRealtime, setLiveRealtime] = useState<RealtimeData | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('selectedVehicle');
    if (saved && VEHICLE_PROFILES[saved]) {
      setCurrentVehicle(saved);
    }
  }, []);

  // Save to localStorage when changed
  useEffect(() => {
    localStorage.setItem('selectedVehicle', currentVehicle);
  }, [currentVehicle]);

  const baseVehicle = VEHICLE_PROFILES[currentVehicle];

  // Reset live data when vehicle changes
  useEffect(() => {
    setLiveRealtime(null);
  }, [currentVehicle]);

  // Live telemetry simulation — fluctuates realtime values every 3s
  useEffect(() => {
    const base = baseVehicle.realtime;
    const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

    const id = setInterval(() => {
      setLiveRealtime((prev) => {
        const r = prev ?? { ...base };
        return {
          ...r,
          power_draw_kw: clamp(
            r.power_draw_kw + (Math.random() - 0.5) * 0.8,
            base.power_draw_kw * 0.7,
            base.power_draw_kw * 1.3
          ),
          motor_temp_c: clamp(
            r.motor_temp_c + (Math.random() - 0.5) * 0.6,
            base.motor_temp_c - 4,
            base.motor_temp_c + 4
          ),
          inverter_temp_c: clamp(
            r.inverter_temp_c + (Math.random() - 0.5) * 0.4,
            base.inverter_temp_c - 3,
            base.inverter_temp_c + 3
          ),
          coolant_temp_c: clamp(
            r.coolant_temp_c + (Math.random() - 0.5) * 0.3,
            base.coolant_temp_c - 2,
            base.coolant_temp_c + 2
          ),
          efficiency_wh_per_km: clamp(
            r.efficiency_wh_per_km + (Math.random() - 0.5) * 3,
            base.efficiency_wh_per_km - 8,
            base.efficiency_wh_per_km + 8
          ),
          cabin_hvac_kw: r.cabin_hvac_kw,
          auxiliary_kw: r.auxiliary_kw,
          optimal_wh_per_km: r.optimal_wh_per_km,
          regen_active: r.regen_active,
        };
      });
    }, 3000);

    return () => clearInterval(id);
  }, [baseVehicle]);

  // Merge live realtime data onto the base profile
  const vehicle: VehicleProfile = liveRealtime
    ? { ...baseVehicle, realtime: liveRealtime }
    : baseVehicle;

  const switchVehicle = (vehicleId: string) => {
    if (VEHICLE_PROFILES[vehicleId]) {
      setCurrentVehicle(vehicleId);
    }
  };

  const value = {
    vehicle,
    currentVehicle,
    switchVehicle,
    allVehicles: VEHICLE_PROFILES,
  };

  return (
    <VehicleContext.Provider value={value}>
      {children}
    </VehicleContext.Provider>
  );
}

export function useVehicle() {
  const context = useContext(VehicleContext);
  if (!context) {
    throw new Error('useVehicle must be used within VehicleProvider');
  }
  return context;
}

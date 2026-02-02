import sys
import os

print(f"CWD: {os.getcwd()}")
try:
    from app.simulation.converters import BatteryEnergyConverter
    print("Import BatteryEnergyConverter: SUCCESS")
except Exception as e:
    print(f"Import BatteryEnergyConverter: FAILED - {e}")

try:
    from app.simulation.efficiency import EfficiencyModel
    print("Import EfficiencyModel: SUCCESS")
except Exception as e:
    print(f"Import EfficiencyModel: FAILED - {e}")

try:
    from app.simulation.schemas import VehicleParameters, SimulationStep
    print("Import Schemas: SUCCESS")
except Exception as e:
    print(f"Import Schemas: FAILED - {e}")

try:
    from app.simulation.battery_model import BatteryModel
    print("Import BatteryModel: SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Import BatteryModel: FAILED - {e}")

try:
    from app.simulation.energy_calculator import EnergyCalculator
    print("Import EnergyCalculator: SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Import EnergyCalculator: FAILED - {e}")

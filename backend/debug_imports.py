import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    from app.simulation.converters import BatteryEnergyConverter
    print("Import BatteryEnergyConverter: SUCCESS")
except ImportError as e:
    print(f"Import BatteryEnergyConverter: FAILED - {e}")

try:
    from app.simulation.efficiency import EfficiencyModel
    print("Import EfficiencyModel: SUCCESS")
except ImportError as e:
    print(f"Import EfficiencyModel: FAILED - {e}")

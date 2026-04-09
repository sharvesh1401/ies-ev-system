from pathlib import Path
import sys

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    from app.config import settings
    print("SUCCESS: Config loaded!")
    print(f"ORS_API_KEY: {'[SET]' if settings.ORS_API_KEY else '[MISSING]'}")
    print(f"OPENWEATHER_API_KEY: {'[SET]' if settings.OPENWEATHER_API_KEY else '[MISSING]'}")
    print(f"OPENCHARGE_API_KEY: {'[SET]' if settings.OPENCHARGE_API_KEY else '[MISSING]'}")
    print(f"DEEPSEEK_API_KEY: {'[SET]' if settings.DEEPSEEK_API_KEY else '[MISSING]'}")
except Exception as e:
    print(f"ERROR: Failed to load config: {e}")

# Configuration module
# Re-export settings so 'from app.config import settings' works
# regardless of whether the config package or config.py is resolved.
import importlib.util
import os

# Load app/config.py directly (it's shadowed by this directory)
_config_file = os.path.join(os.path.dirname(__file__), '..', 'config.py')
if os.path.exists(_config_file):
    _spec = importlib.util.spec_from_file_location("app._config_module", _config_file)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    settings = _mod.settings
    Settings = _mod.Settings

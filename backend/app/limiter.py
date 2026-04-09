"""
Shared rate-limiter instance.

Defined here (not in main.py) so route modules can import it without
creating a circular import with app.main.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Default: 100 requests per minute per IP.
# Override per-endpoint with @limiter.limit("N/minute").
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

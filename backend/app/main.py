from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.config import settings
from app.limiter import limiter
from app.routes import health_router, ai_router, simulation_router
from app.routes.prediction_v2 import router as prediction_v2_router
from app.api.routes import predict
import logging
import sentry_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meridian.startup")

# ── Sentry error tracking ────────────────────────────────────────────────────
# Initialise before anything else so all exceptions are captured.
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        # Capture 10% of transactions for performance monitoring (free-tier friendly)
        traces_sample_rate=0.1,
        # Don't send PII like IP addresses or user agents
        send_default_pii=False,
    )
    logger.info("Sentry error tracking enabled")

# limiter is imported from app.limiter to avoid circular imports with route modules.


def _key_status(value: str | None, name: str) -> str:
    if not value or value.startswith("your_"):
        return f"  ✗ {name}: NOT CONFIGURED"
    return f"  ✓ {name}: configured ({value[:6]}...)"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: log external API key status ──────────────────────────────
    logger.info("=" * 56)
    logger.info("  Meridian IES-EV Backend  —  External API Status")
    logger.info("=" * 56)
    logger.info(_key_status(settings.ORS_API_KEY, "OpenRouteService  (ORS_API_KEY)"))
    logger.info(_key_status(settings.OPENWEATHER_API_KEY, "OpenWeatherMap    (OPENWEATHER_API_KEY)"))
    logger.info(_key_status(settings.OPENCHARGE_API_KEY, "Open Charge Map   (OPENCHARGE_API_KEY)"))
    logger.info("  ✓ Open-Elevation: no key required")
    logger.info(_key_status(settings.DEEPSEEK_API_KEY, "DeepSeek          (DEEPSEEK_API_KEY)"))
    logger.info(_key_status(settings.GROQ_API_KEY, "Groq              (GROQ_API_KEY)"))
    logger.info(_key_status(settings.GEMINI_API_KEY, "Gemini            (GEMINI_API_KEY)"))
    missing = [
        n for n, v in [
            ("ORS_API_KEY", settings.ORS_API_KEY),
            ("OPENWEATHER_API_KEY", settings.OPENWEATHER_API_KEY),
            ("OPENCHARGE_API_KEY", settings.OPENCHARGE_API_KEY),
        ]
        if not v or v.startswith("your_")
    ]
    if missing:
        logger.warning(
            f"  ⚠  {len(missing)} key(s) not set. "
            "Route Planner features will return errors until configured in .env"
        )
    else:
        logger.info("  All external keys present — run GET /api/external/health to verify")
    logger.info("=" * 56)
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for IES_EV Project",
    lifespan=lifespan,
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Attach the limiter to app state so @limiter.limit() decorators can find it.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS ──────────────────────────────────────────────────────────────────────
# OWASP: never combine allow_credentials=True with a wildcard origin.
# Only allow the origins we actually serve from.
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # Vite dev server / Docker frontend
    "http://127.0.0.1:3000",
    "http://localhost:5173",   # Local Vite
    "http://localhost:8080",   # Adminer (internal tooling)
    "http://127.0.0.1:8080",
    "https://meridian-ev.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Security headers ──────────────────────────────────────────────────────────
# OWASP: send defensive headers on every response.
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        return response

app.add_middleware(SecurityHeadersMiddleware)

from app.routes.external_apis import router as external_api_router

# Include Routers
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI"])
app.include_router(simulation_router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(prediction_v2_router, prefix="/api/v2", tags=["Prediction v2"])
app.include_router(external_api_router, prefix="/api/external", tags=["External Proxies"])
app.include_router(predict.router)

# Monitoring
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    """
    Root endpoint to verify service is running.
    """
    return {
        "message": "Welcome to IES_EV API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

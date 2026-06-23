import logging

logger = logging.getLogger(__name__)

# Lazy database and redis — gracefully handle missing dependencies
engine = None
SessionLocal = None
Base = None

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker, declarative_base
    from app.config import settings
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.warning(f"Database not available (psycopg2 or PostgreSQL not configured): {e}")

def get_db():
    if SessionLocal is None:
        raise RuntimeError("Database not configured — PostgreSQL not available")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis
def get_redis_client():
    try:
        import redis
        from app.config import settings
        return redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=5
        )
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        return None

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.core.config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    # Import all ORM models before create_all so SQLAlchemy registers every table.
    from backend.database.models.user import User  # noqa: F401
    from backend.database.models.asset import Asset  # noqa: F401
    from backend.database.models.market_data import MarketData  # noqa: F401
    from backend.database.models.features import Features  # noqa: F401
    from backend.database.models.market_regime import MarketRegime  # noqa: F401
    from backend.database.models.asset_universe import AssetUniverse  # noqa: F401

    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)

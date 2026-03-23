from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings


engine = create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_users_table() -> None:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    columns = {column["name"]: column for column in inspector.get_columns("users")}
    with engine.begin() as connection:
        id_type = str(columns["id"]["type"]).upper()
        if "INT" in id_type:
            connection.execute(text("ALTER TABLE users ALTER COLUMN id DROP DEFAULT"))
            connection.execute(text("ALTER TABLE users ALTER COLUMN id TYPE VARCHAR USING id::text"))

        if "full_name" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN full_name VARCHAR DEFAULT ''"))
        if "password_hash" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR"))
        if "risk_level" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN risk_level VARCHAR"))
        if "goals" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN goals VARCHAR"))
        if "investment_horizon" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN investment_horizon VARCHAR"))
        if "interests" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN interests JSON DEFAULT '[]'::json"))
        if "onboarding_complete" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN onboarding_complete BOOLEAN DEFAULT FALSE"))

        connection.execute(
            text(
                """
                UPDATE users
                SET full_name = COALESCE(NULLIF(full_name, ''), ''),
                    password_hash = COALESCE(NULLIF(password_hash, ''), 'legacy-user-needs-reset'),
                    risk_level = COALESCE(NULLIF(risk_level, ''), risk_tolerance, 'medium'),
                    goals = COALESCE(NULLIF(goals, ''), investment_goal, 'balanced growth'),
                    investment_horizon = COALESCE(NULLIF(investment_horizon, ''), '5y'),
                    capital = COALESCE(capital, 0.0),
                    interests = COALESCE(interests, '[]'::json),
                    onboarding_complete = COALESCE(onboarding_complete, FALSE)
                """
            )
        )


def _migrate_portfolios_table() -> None:
    inspector = inspect(engine)
    if "portfolios" not in inspector.get_table_names():
        return

    columns = {column["name"]: column for column in inspector.get_columns("portfolios")}
    with engine.begin() as connection:
        if "positions_json" not in columns:
            connection.execute(text("ALTER TABLE portfolios ADD COLUMN positions_json JSON DEFAULT '[]'::json"))
        if "lookback_days" not in columns:
            connection.execute(text("ALTER TABLE portfolios ADD COLUMN lookback_days INTEGER DEFAULT 180"))

        connection.execute(
            text(
                """
                UPDATE portfolios
                SET positions_json = COALESCE(positions_json, '[]'::json),
                    lookback_days = COALESCE(lookback_days, 180),
                    strategy = COALESCE(strategy, ''),
                    equity_pct = COALESCE(equity_pct, 0.0),
                    cash_pct = COALESCE(cash_pct, 0.0)
                """
            )
        )


def create_tables():
    from backend.db.models import FeaturesLatest, FinancialPlan, MemoryMessage, Portfolio, User  # noqa: F401
    from backend.database.models.all_assets import AllAssets  # noqa: F401
    from backend.database.models.asset import Asset  # noqa: F401
    from backend.database.models.asset_universe import AssetUniverse  # noqa: F401
    from backend.database.models.features import Features  # noqa: F401
    from backend.database.models.features_latest import FeaturesLatest as FeaturesLatestAlias  # noqa: F401
    from backend.database.models.market_data import MarketData  # noqa: F401
    from backend.database.models.market_regime import MarketRegime  # noqa: F401

    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    _migrate_users_table()
    _migrate_portfolios_table()

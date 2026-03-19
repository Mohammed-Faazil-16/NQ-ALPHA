from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.market_data_agent.fetch_crypto import fetch_crypto_data
from agents.market_data_agent.fetch_macro import fetch_macro_data
from agents.market_data_agent.fetch_stocks import fetch_stock_data
from backend.database.models.asset import Asset
from backend.database.models.market_data import MarketData
from backend.database.postgres import SessionLocal


DEFAULT_STOCK_START = datetime(2015, 1, 1)


def update_market_data():
    db = SessionLocal()
    try:
        assets = db.query(Asset).filter(Asset.active.is_(True)).all()

        for asset in assets:
            if asset.asset_type == "stock":
                last_row = (
                    db.query(MarketData)
                    .filter(MarketData.symbol == asset.symbol)
                    .order_by(MarketData.timestamp.desc())
                    .first()
                )
                if last_row:
                    start_date = (last_row.timestamp + timedelta(days=1)).date().isoformat()
                else:
                    start_date = DEFAULT_STOCK_START.date().isoformat()
                end_date = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
                fetch_stock_data(asset.symbol, start_date, end_date)
            elif asset.asset_type == "crypto":
                fetch_crypto_data(asset.symbol)
            elif asset.asset_type == "macro":
                fetch_macro_data(asset.symbol)
    finally:
        db.close()


if __name__ == "__main__":
    update_market_data()

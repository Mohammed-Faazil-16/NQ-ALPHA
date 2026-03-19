from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.asset import Asset
from backend.database.postgres import SessionLocal


SAMPLE_ASSETS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "asset_type": "stock", "exchange": "NASDAQ", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "asset_type": "stock", "exchange": "NASDAQ", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "asset_type": "stock", "exchange": "NASDAQ", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc.", "asset_type": "stock", "exchange": "NASDAQ", "sector": "Consumer Discretionary"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "asset_type": "stock", "exchange": "NASDAQ", "sector": "Communication Services"},
    {"symbol": "META", "name": "Meta Platforms, Inc.", "asset_type": "stock", "exchange": "NASDAQ", "sector": "Communication Services"},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "asset_type": "stock", "exchange": "NASDAQ", "sector": "Consumer Discretionary"},
    {"symbol": "BTC/USDT", "name": "Bitcoin / Tether", "asset_type": "crypto", "exchange": "Binance", "sector": None},
    {"symbol": "ETH/USDT", "name": "Ethereum / Tether", "asset_type": "crypto", "exchange": "Binance", "sector": None},
    {"symbol": "SOL/USDT", "name": "Solana / Tether", "asset_type": "crypto", "exchange": "Binance", "sector": None},
    {"symbol": "BNB/USDT", "name": "BNB / Tether", "asset_type": "crypto", "exchange": "Binance", "sector": None},
    {"symbol": "XRP/USDT", "name": "XRP / Tether", "asset_type": "crypto", "exchange": "Binance", "sector": None},
    {"symbol": "GC=F", "name": "Gold Futures", "asset_type": "commodity", "exchange": "COMEX", "sector": None},
    {"symbol": "SI=F", "name": "Silver Futures", "asset_type": "commodity", "exchange": "COMEX", "sector": None},
    {"symbol": "CL=F", "name": "Crude Oil Futures", "asset_type": "commodity", "exchange": "NYMEX", "sector": None},
    {"symbol": "FEDFUNDS", "name": "Federal Funds Effective Rate", "asset_type": "macro", "exchange": None, "sector": None},
    {"symbol": "CPIAUCSL", "name": "Consumer Price Index for All Urban Consumers", "asset_type": "macro", "exchange": None, "sector": None},
    {"symbol": "UNRATE", "name": "Unemployment Rate", "asset_type": "macro", "exchange": None, "sector": None},
]


def load_assets():
    db = SessionLocal()
    try:
        for asset_data in SAMPLE_ASSETS:
            existing_asset = db.query(Asset).filter(Asset.symbol == asset_data["symbol"]).first()
            if existing_asset:
                print(f"Skipped: {asset_data['symbol']}")
                continue

            db.add(Asset(**asset_data))
            print(f"Inserted: {asset_data['symbol']}")

        db.commit()
        print("Asset loading complete")
    finally:
        db.close()


if __name__ == "__main__":
    load_assets()

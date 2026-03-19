from pathlib import Path
import re
import sys

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.asset import Asset
from backend.database.postgres import SessionLocal


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TICKER_PATTERN = re.compile(r"^[A-Z0-9\-=/]+$")

NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "ADBE","NFLX","PEP","CSCO","AMD","TMUS","INTC","QCOM","TXN","AMAT",
    "INTU","ISRG","BKNG","ADP","MDLZ","LRCX","MU","PANW","SNPS","CDNS",
    "KLAC","CRWD","FTNT","WDAY","ADSK","ORLY","REGN","VRTX","DXCM","MELI",
    "KDP","IDXX","PAYX","MRNA","TEAM","GILD","CTAS","ODFL","MNST","CSX",
    "NXPI","PCAR","ROST","CPRT","FANG","ANSS","EXC","FAST","BIIB","SIRI",
    "DLTR","VRSK","KHC","AEP","BKR","XEL","AZN","EA","LULU","WBD",
    "GEHC","ILMN","ALGN","MAR","MCHP","ABNB","CHTR","MRVL","ADI","PDD",
    "DDOG","ZS","OKTA","DOCU","NET","MDB","SNOW","RIVN","LCID","PLTR",
    "HOOD","COIN","ARM","SMCI","TTD","ON","GFS","ENPH","AMGN","CSGP"
]

CRYPTO = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","BNB/USDT","ADA/USDT","DOGE/USDT","AVAX/USDT","DOT/USDT","LINK/USDT",
    "TRX/USDT","MATIC/USDT","LTC/USDT","BCH/USDT","ATOM/USDT","ETC/USDT","XLM/USDT","NEAR/USDT","APT/USDT","ARB/USDT",
    "OP/USDT","FIL/USDT","ICP/USDT","INJ/USDT","RNDR/USDT","SUI/USDT","SEI/USDT","PEPE/USDT","SHIB/USDT","UNI/USDT",
    "AAVE/USDT","MKR/USDT","SNX/USDT","CRV/USDT","LDO/USDT","DYDX/USDT","TIA/USDT","JUP/USDT","PYTH/USDT","WIF/USDT",
    "FET/USDT","AGIX/USDT","OCEAN/USDT","GRT/USDT","IMX/USDT","SAND/USDT","MANA/USDT","AXS/USDT","RUNE/USDT","EGLD/USDT",
    "KAS/USDT","STX/USDT","TON/USDT","NOT/USDT","ONDO/USDT","ENA/USDT","TAO/USDT","AR/USDT","GMX/USDT","COMP/USDT",
    "SUSHI/USDT","1INCH/USDT","ZEC/USDT","DASH/USDT","NEO/USDT","QTUM/USDT","IOTA/USDT","ALGO/USDT","VET/USDT","THETA/USDT",
    "FLOW/USDT","EOS/USDT","XTZ/USDT","KAVA/USDT","ROSE/USDT","CELO/USDT","CHZ/USDT","FTM/USDT","HBAR/USDT","MINA/USDT",
    "KSM/USDT","ENS/USDT","BAL/USDT","YFI/USDT","ANKR/USDT","API3/USDT","WLD/USDT","BLUR/USDT","ACH/USDT","ASTR/USDT",
    "BOME/USDT","BONK/USDT","CFX/USDT","CKB/USDT","DYM/USDT","JASMY/USDT","LRC/USDT","MASK/USDT","MEME/USDT","ORDI/USDT"
]

COMMODITIES = [
    "GC=F","SI=F","CL=F","NG=F","HG=F","PL=F","PA=F",
    "ZC=F","ZW=F","ZS=F","KC=F","SB=F","CC=F","CT=F",
    "OJ=F","RB=F","HO=F","BZ=F","LE=F","HE=F"
]

MACRO = [
    "FEDFUNDS","CPIAUCSL","UNRATE","GDP","PCE",
    "M2SL","INDPRO","PAYEMS","DGS10","DGS2",
    "UMCSENT","HOUST","ICSA","CP","DEXUSEU",
    "DEXUSUK","DEXJPUS","DEXCHUS","DTB3","T10Y2Y"
]

FX = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X",
    "USDCHF=X","NZDUSD=X","EURGBP=X","EURJPY=X","GBPJPY=X",
    "AUDJPY=X","CHFJPY=X","EURCAD=X","EURAUD=X","EURNZD=X",
    "GBPCAD=X","GBPAUD=X","AUDCAD=X","AUDNZD=X","CADJPY=X"
]

MACRO_NAMES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
    "PCE": "Personal Consumption Expenditures",
    "M2SL": "M2 Money Stock",
    "INDPRO": "Industrial Production Index",
    "PAYEMS": "Total Nonfarm Payrolls",
    "DGS10": "10-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "UMCSENT": "Michigan Consumer Sentiment",
    "HOUST": "Housing Starts",
    "ICSA": "Initial Claims",
    "CP": "Corporate Profits",
    "DEXUSEU": "USD to Euro FX Rate",
    "DEXUSUK": "USD to British Pound FX Rate",
    "DEXJPUS": "Japanese Yen to USD FX Rate",
    "DEXCHUS": "Chinese Yuan to USD FX Rate",
    "DTB3": "3-Month Treasury Bill",
    "T10Y2Y": "10Y-2Y Treasury Spread",
}

COMMODITY_NAMES = {
    "GC=F": "Gold Futures",
    "SI=F": "Silver Futures",
    "CL=F": "Crude Oil Futures",
    "NG=F": "Natural Gas Futures",
    "HG=F": "Copper Futures",
    "PL=F": "Platinum Futures",
    "PA=F": "Palladium Futures",
    "ZC=F": "Corn Futures",
    "ZW=F": "Wheat Futures",
    "ZS=F": "Soybean Futures",
    "KC=F": "Coffee Futures",
    "SB=F": "Sugar Futures",
    "CC=F": "Cocoa Futures",
    "CT=F": "Cotton Futures",
    "OJ=F": "Orange Juice Futures",
    "RB=F": "RBOB Gasoline Futures",
    "HO=F": "Heating Oil Futures",
    "BZ=F": "Brent Crude Oil Futures",
    "LE=F": "Live Cattle Futures",
    "HE=F": "Lean Hogs Futures",
}

FX_NAMES = {
    "EURUSD=X": "Euro / US Dollar",
    "GBPUSD=X": "British Pound / US Dollar",
    "USDJPY=X": "US Dollar / Japanese Yen",
    "AUDUSD=X": "Australian Dollar / US Dollar",
    "USDCAD=X": "US Dollar / Canadian Dollar",
    "USDCHF=X": "US Dollar / Swiss Franc",
    "NZDUSD=X": "New Zealand Dollar / US Dollar",
    "EURGBP=X": "Euro / British Pound",
    "EURJPY=X": "Euro / Japanese Yen",
    "GBPJPY=X": "British Pound / Japanese Yen",
    "AUDJPY=X": "Australian Dollar / Japanese Yen",
    "CHFJPY=X": "Swiss Franc / Japanese Yen",
    "EURCAD=X": "Euro / Canadian Dollar",
    "EURAUD=X": "Euro / Australian Dollar",
    "EURNZD=X": "Euro / New Zealand Dollar",
    "GBPCAD=X": "British Pound / Canadian Dollar",
    "GBPAUD=X": "British Pound / Australian Dollar",
    "AUDCAD=X": "Australian Dollar / Canadian Dollar",
    "AUDNZD=X": "Australian Dollar / New Zealand Dollar",
    "CADJPY=X": "Canadian Dollar / Japanese Yen",
}


def load_sp500_assets():
    print("Fetching S&P500 asset universe...")

    headers = {
        "User-Agent": "Mozilla/5.0 (NeuroQuant Data Collector)"
    }

    response = requests.get(SP500_URL, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})

    if table is None:
        raise RuntimeError("S&P500 constituents table not found on Wikipedia")

    tbody = table.find("tbody")
    if tbody is None:
        raise RuntimeError("S&P500 constituents table body not found on Wikipedia")

    rows = tbody.find_all("tr")
    assets = []
    seen_symbols = set()

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        symbol = cols[0].text.strip().replace(".", "-")
        name = cols[1].text.strip()
        sector = cols[2].text.strip()

        if not symbol or not TICKER_PATTERN.fullmatch(symbol) or symbol in seen_symbols:
            continue

        assets.append(
            {
                "symbol": symbol,
                "name": name,
                "asset_type": "stock",
                "exchange": "NYSE/NASDAQ",
                "sector": sector,
                "active": True,
            }
        )
        seen_symbols.add(symbol)

    if len(assets) < 400:
        raise RuntimeError("Parsed S&P500 list is suspiciously small")

    print(f"S&P500 loaded: {len(assets)}")
    return assets


def load_nasdaq_assets():
    assets = [
        {
            "symbol": symbol,
            "name": symbol,
            "asset_type": "stock",
            "exchange": "NASDAQ",
            "sector": None,
            "active": True,
        }
        for symbol in NASDAQ100
    ]
    print(f"NASDAQ100 loaded: {len(assets)}")
    return assets


def load_crypto_assets():
    assets = [
        {
            "symbol": symbol,
            "name": symbol,
            "asset_type": "crypto",
            "exchange": "crypto",
            "sector": None,
            "active": True,
        }
        for symbol in CRYPTO
    ]
    print(f"Crypto loaded: {len(assets)}")
    return assets


def load_commodity_assets():
    assets = [
        {
            "symbol": symbol,
            "name": COMMODITY_NAMES.get(symbol, symbol),
            "asset_type": "commodity",
            "exchange": "Yahoo Finance",
            "sector": None,
            "active": True,
        }
        for symbol in COMMODITIES
    ]
    print(f"Commodities loaded: {len(assets)}")
    return assets


def load_macro_assets():
    assets = [
        {
            "symbol": symbol,
            "name": MACRO_NAMES.get(symbol, symbol),
            "asset_type": "macro",
            "exchange": "FRED",
            "sector": None,
            "active": True,
        }
        for symbol in MACRO
    ]
    print(f"Macro loaded: {len(assets)}")
    return assets


def load_fx_assets():
    assets = [
        {
            "symbol": symbol,
            "name": FX_NAMES.get(symbol, symbol),
            "asset_type": "fx",
            "exchange": "Yahoo Finance",
            "sector": None,
            "active": True,
        }
        for symbol in FX
    ]
    print(f"FX loaded: {len(assets)}")
    return assets


def insert_assets(db, asset_payloads):
    if not asset_payloads:
        return 0, 0

    deduped_assets = {}
    for asset in asset_payloads:
        symbol = asset.get("symbol")
        if symbol and TICKER_PATTERN.fullmatch(symbol):
            deduped_assets[symbol] = asset

    symbols = list(deduped_assets.keys())
    existing_symbols = {
        row[0] for row in db.query(Asset.symbol).filter(Asset.symbol.in_(symbols)).all()
    }

    new_assets = [
        Asset(**asset)
        for symbol, asset in deduped_assets.items()
        if symbol not in existing_symbols
    ]

    if new_assets:
        db.bulk_save_objects(new_assets)
        db.commit()

    return len(new_assets), len(existing_symbols)


def main():
    print("Loading asset universe...")

    db = SessionLocal()
    try:
        loaders = [
            load_sp500_assets,
            load_nasdaq_assets,
            load_crypto_assets,
            load_commodity_assets,
            load_macro_assets,
            load_fx_assets,
        ]

        total_processed = 0
        total_inserted = 0
        total_existing = 0

        for loader in loaders:
            assets = loader()
            inserted, existing = insert_assets(db, assets)
            total_processed += len(assets)
            total_inserted += inserted
            total_existing += existing

        print(f"Total assets processed: {total_processed}")
        print(f"New assets inserted: {total_inserted}")
        print(f"Assets already existing: {total_existing}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

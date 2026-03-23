from backend.services.asset_ingestion_service import ingest_assets


if __name__ == "__main__":
    result = ingest_assets()
    print(result["message"])

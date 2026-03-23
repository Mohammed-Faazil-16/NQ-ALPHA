from sqlalchemy import Column, Integer, String

from backend.database.postgres import Base


class AllAssets(Base):
    __tablename__ = "all_assets"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    asset_type = Column(String, nullable=False, default="stock")

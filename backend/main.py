from fastapi import FastAPI

from backend.api.user_routes import router as user_router
from backend.database.postgres import create_tables


app = FastAPI()

app.include_router(user_router, prefix="/users", tags=["users"])

@app.on_event("startup")
def startup():
    create_tables()



@app.get("/")
def read_root():
    return {"message": "NeuroQuant backend running"}


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "NeuroQuant backend healthy"}

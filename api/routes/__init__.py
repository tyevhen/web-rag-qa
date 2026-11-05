from fastapi import APIRouter

from api.routes import ask, corpus, health, ingest

router = APIRouter()
for feature in (ask, ingest, corpus, health):
    router.include_router(feature.router)

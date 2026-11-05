from fastapi import APIRouter

from api.deps import DocumentsDep
from api.schemas import DocumentSummary

router = APIRouter()


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(repo: DocumentsDep) -> list[DocumentSummary]:
    rows = await repo.list_documents()
    return [
        DocumentSummary(
            id=row["id"],
            source=row["filename"],
            chunks=row["chunks"],
            ingested_at=row["uploaded_at"].isoformat(),
        )
        for row in rows
    ]

import pytest
import httpx


@pytest.fixture
async def client():
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=120.0) as c:
        yield c

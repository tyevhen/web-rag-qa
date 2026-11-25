from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.errors import http_error_handler
from api.middleware import RequestContextMiddleware

app = FastAPI()
app.add_exception_handler(StarletteHTTPException, http_error_handler)  # type: ignore[arg-type]
app.add_middleware(RequestContextMiddleware)


@app.get("/ok")
async def ok(request: Request) -> dict[str, str]:
    return {"run_id": request.state.run_id}


@app.get("/missing")
async def missing() -> None:
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/boom")
async def boom() -> None:
    raise RuntimeError("bug")


client = TestClient(app, raise_server_exceptions=False)


def test_the_route_sees_the_id_it_returns_in_the_header():
    resp = client.get("/ok")
    assert resp.json()["run_id"] == resp.headers["x-request-id"]


def test_http_errors_and_unknown_paths_share_the_error_body():
    for path in ("/missing", "/nope"):
        resp = client.get(path)
        assert resp.status_code == 404 and resp.json()["run_id"] == resp.headers["x-request-id"]
    assert client.get("/missing").json()["detail"] == "Job not found"


def test_an_unhandled_error_is_a_500_with_the_request_id():
    resp = client.get("/boom")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error", "run_id": resp.headers["x-request-id"]}

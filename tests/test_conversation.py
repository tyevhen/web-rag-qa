"""
Integration tests for /ask input validation and server-side conversation history.
Requires a running API (uvicorn api.app:app); the last test also needs ingested content.
"""


async def test_question_over_limit_is_rejected(client):
    resp = await client.post("/ask", json={"question": "x" * 2001})
    assert resp.status_code == 422


async def test_client_history_is_rejected(client):
    resp = await client.post("/ask", json={
        "question": "What should I keep in an emergency kit?",
        "history": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 422
    assert any(e["loc"][-1] == "history" for e in resp.json()["detail"])


async def test_unknown_model_is_rejected(client):
    models = (await client.get("/models")).json()
    assert models["default"] == models["models"][0]
    resp = await client.post("/ask", json={"question": "hi", "model": "no-such-model"})
    assert resp.status_code == 422


async def test_unknown_chunk_is_404(client):
    assert (await client.get("/chunks/00000000-0000-0000-0000-000000000000")).status_code == 404


async def test_same_conversation_id_continues_conversation(client):
    r1 = await client.post("/ask", json={"question": "What are the main items I should keep in an emergency kit?"})
    assert r1.status_code == 200
    conversation_id = r1.json()["conversation_id"]
    assert conversation_id

    r2 = await client.post("/ask", json={
        "question": "How long should each of those supplies last?",
        "conversation_id": conversation_id,
    })
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conversation_id

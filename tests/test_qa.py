"""
Integration tests for the /ask endpoint.
Requires a running API (uvicorn api.app:app) with content already ingested.
"""


async def test_complete_question_returns_answer(client):
    resp = await client.post("/ask", json={
        "question": (
            "What items should I include in a 72-hour emergency preparedness "
            "kit for a family of four, and how should I store them?"
        ),
        "history": [],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert "wasn't able" not in data["answer"]
    assert data["clarifying_questions"] == []


async def test_out_of_domain_question_returns_clarifications(client):
    resp = await client.post("/ask", json={
        "question": "How do I fix my bike chain?",
        "history": [],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["clarifying_questions"], "Expected clarifying questions when corpus has no relevant content"
    assert "wasn't able" in data["answer"]


async def test_multi_turn_conversation(client):
    # Turn 1: specific question — agent should answer directly
    r1 = await client.post("/ask", json={
        "question": "What are the main items I should keep in an emergency kit?",
        "history": [],
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["answer"]
    assert d1["clarifying_questions"] == []

    # Turn 2: follow-up that relies on prior context passed via history
    r2 = await client.post("/ask", json={
        "question": "How long should each of those supplies last?",
        "history": [
            {"role": "user", "content": "What are the main items I should keep in an emergency kit?"},
            {"role": "assistant", "content": d1["answer"]},
        ],
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["answer"]
    assert "wasn't able" not in d2["answer"]
    assert d2["clarifying_questions"] == []

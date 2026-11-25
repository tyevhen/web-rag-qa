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
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert "wasn't able" not in data["answer"]
    assert data["clarifying_questions"] == []


async def test_out_of_domain_question_returns_clarifications(client):
    resp = await client.post("/ask", json={
        "question": "How do I fix my bike chain?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["clarifying_questions"], "Expected clarifying questions when corpus has no relevant content"
    assert "wasn't able" in data["answer"]


async def test_multi_turn_conversation(client):
    # Turn 1: specific question — agent should answer directly
    r1 = await client.post("/ask", json={
        "question": "What are the main items I should keep in an emergency kit?",
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["answer"]
    assert d1["clarifying_questions"] == []

    # Turn 2: follow-up that relies on the server-side history of turn 1
    r2 = await client.post("/ask", json={
        "question": "How long should each of those supplies last?",
        "conversation_id": d1["conversation_id"],
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["answer"]
    assert "wasn't able" not in d2["answer"]
    assert d2["clarifying_questions"] == []

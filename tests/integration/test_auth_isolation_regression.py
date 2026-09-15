"""Cross-user authentication and data isolation regression test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.deps import get_db
from app.main import app
from app.models.journal import JournalEntry


@pytest.fixture(name="api_client")
def api_client_fixture(session: Session):
    """Provide TestClient with test database session override."""

    def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestCrossUserIsolationRegression:
    """Rigorous verification that user boundaries cannot be traversed under any circumstances."""

    def test_complete_cross_user_isolation(self, api_client: TestClient, session: Session):
        # 1. Register and login User Alice
        resp_reg_a = api_client.post(
            "/api/v1/auth/register",
            json={"email": "alice_iso@example.com", "password": "Password123!@#"},
        )
        assert resp_reg_a.status_code == 201
        user_a_id = str(resp_reg_a.json()["id"])

        resp_login_a = api_client.post(
            "/api/v1/auth/login",
            json={"email": "alice_iso@example.com", "password": "Password123!@#"},
        )
        assert resp_login_a.status_code == 200
        token_a = resp_login_a.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        # 2. Register and login User Bob
        resp_reg_b = api_client.post(
            "/api/v1/auth/register",
            json={"email": "bob_iso@example.com", "password": "Password123!@#"},
        )
        assert resp_reg_b.status_code == 201

        resp_login_b = api_client.post(
            "/api/v1/auth/login",
            json={"email": "bob_iso@example.com", "password": "Password123!@#"},
        )
        assert resp_login_b.status_code == 200
        token_b = resp_login_b.json()["access_token"]
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 3. Create a journal entry directly for Alice
        alice_entry = JournalEntry(
            user_id=user_a_id,
            text="Alice's confidential journal thought.",
            ai_response="Reflect on this safely.",
        )
        session.add(alice_entry)
        session.commit()
        session.refresh(alice_entry)
        entry_id = str(alice_entry.id)

        # 4. User Bob attempts to access Alice's entry -> MUST return 404 (zero existence leakage)
        get_res = api_client.get(f"/api/v1/journals/{entry_id}", headers=headers_b)
        assert get_res.status_code == 404
        assert "not found" in get_res.json()["detail"].lower()

        # 5. User Bob attempts to update Alice's entry -> MUST return 404
        patch_res = api_client.patch(
            f"/api/v1/journals/{entry_id}",
            json={"tags": ["malicious_tag"]},
            headers=headers_b,
        )
        assert patch_res.status_code == 404

        # 6. User Bob attempts to soft-delete Alice's entry -> MUST return 404
        del_res = api_client.delete(f"/api/v1/journals/{entry_id}", headers=headers_b)
        assert del_res.status_code == 404

        # 7. User Bob attempts to restore Alice's entry -> MUST return 404
        res_restore = api_client.post(f"/api/v1/journals/{entry_id}/restore", headers=headers_b)
        assert res_restore.status_code == 404

        # 8. User Bob attempts to hard-delete Alice's entry -> MUST return 404
        res_hard_del = api_client.post(
            f"/api/v1/journals/{entry_id}/hard-delete", headers=headers_b
        )
        assert res_hard_del.status_code == 404

        # 9. User Bob exports all journals as JSON -> Alice's entry must NOT appear
        res_export = api_client.get("/api/v1/journals/export?format=json", headers=headers_b)
        assert res_export.status_code == 200
        assert len(res_export.json()) == 0

        # 10. Alice can access her own entry without issue
        alice_res = api_client.get(f"/api/v1/journals/{entry_id}", headers=headers_a)
        assert alice_res.status_code == 200
        assert alice_res.json()["text"] == "Alice's confidential journal thought."

"""
tests/test_static_ui.py -- Phase 10b: tests for the static demo UI route
served from api/main.py (GET "/").

Confirms the route serves real HTML (not a 404/500), and that the
static asset mount is wired correctly -- does NOT test visual appearance
or JS behavior, which is out of scope for an automated test.
"""
from fastapi.testclient import TestClient

from api.main import app


def test_root_serves_html():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "EmberRisk" in response.text


def test_root_html_references_model_info_endpoint():
    """Sanity check that the served page actually wires up to the real
    API rather than serving a stale/unrelated file."""
    with TestClient(app) as client:
        response = client.get("/")
        assert "/model/info" in response.text
        assert "/predict" in response.text

import os
import requests
import pytest

BASE_URL = os.getenv("IMAGINATOR_BASE_URL") or os.getenv("RENDER_INTERNAL_URL") or "http://127.0.0.1:8000"
API_KEY = os.getenv("IMAGINATOR_AUTH_TOKEN", "testkey")


def _is_service_up(url: str) -> bool:
    try:
        r = requests.get(url, timeout=3)
        return r.status_code < 500
    except requests.RequestException:
        return False


@pytest.mark.integration
def test_analyze_endpoint_accepts_expected_payloads():
    """
    Integration test: call the running Imaginator /analyze endpoint on the active Render resource.

    The test tries a few common payload shapes used by local callers and asserts we get a JSON
    response (HTTP 200). If the service is unreachable the test is skipped so CI can control
    when it runs this against a deployed service.
    """

    if not _is_service_up(BASE_URL):
        pytest.skip(f"Service not reachable at {BASE_URL}")

    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}

    payload_variants = [
        {"resumeText": "Experienced data analyst with Python.", "jobDescription": "Looking for a BI analyst."},
        {"resume_text": "Experienced data analyst with Python.", "job_ad": "Looking for a BI analyst."},
        {"resume_text": "Experienced data analyst with Python.", "jobDescription": "Looking for a BI analyst."},
        {"resumeText": "Experienced data analyst with Python.", "job_ad": "Looking for a BI analyst."},
    ]

    last_exception = None
    for payload in payload_variants:
        try:
            r = requests.post(f"{BASE_URL.rstrip('/')}/analyze", json=payload, headers=headers, timeout=30)
        except Exception as exc:
            last_exception = exc
            continue

        # Accept 200 and 201 as success; keep trying other variants on 422 (validation)
        if r.status_code in (200, 201):
            # Should be JSON; if not, at least ensure we have a body
            try:
                data = r.json()
            except Exception:
                pytest.fail(f"Analyze returned non-JSON response (status {r.status_code}): {r.text[:200]}")

            assert data is not None
            return

        # If validation error, try next payload
        if r.status_code == 422:
            continue

        # For other failures keep last response
        pytest.fail(f"Analyze endpoint returned unexpected status {r.status_code}: {r.text[:400]}")

    if last_exception:
        pytest.fail(f"All payload variants failed, last exception: {last_exception}")
    pytest.fail("No payload variant produced a successful response (likely validation mismatch)")

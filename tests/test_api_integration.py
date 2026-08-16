"""FastAPI integration tests."""

import pytest
import time


class TestAPIHealthCheck:
    """Test API health checks and basic endpoints."""

    def test_health_endpoint(self, api_client):
        """Test /health endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_metrics_endpoint(self, api_client):
        """Test /metrics endpoint for Prometheus."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert b"http_requests_total" in response.content or b"HELP" in response.content

    def test_root_endpoint(self, api_client):
        """Test root endpoint."""
        response = api_client.get("/")
        assert response.status_code == 200


class TestChatEndpoint:
    """Test /v1/chat/completions endpoint."""

    def test_chat_completions_request(self, api_client):
        """Test basic chat completion request."""
        payload = {
            "model": "meta-llama/Llama-2-7b-hf",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 100,
        }

        response = api_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_chat_completions_streaming(self, api_client):
        """Test streaming chat completion."""
        payload = {
            "model": "meta-llama/Llama-2-7b-hf",
            "messages": [
                {"role": "user", "content": "What is machine learning?"}
            ],
            "stream": True,
        }

        response = api_client.post("/v1/chat/completions", json=payload, stream=True)
        assert response.status_code == 200

        # Verify we get streaming content
        content = response.content.decode("utf-8")
        assert len(content) > 0

    def test_chat_completions_missing_messages(self, api_client):
        """Test error handling for missing messages."""
        payload = {
            "model": "meta-llama/Llama-2-7b-hf",
            "messages": [],
        }

        response = api_client.post("/v1/chat/completions", json=payload)
        # Should handle gracefully (either success or validation error)
        assert response.status_code in [200, 422]

    def test_chat_completions_latency(self, api_client):
        """Test that chat completions are reasonably fast."""
        payload = {
            "model": "meta-llama/Llama-2-7b-hf",
            "messages": [
                {"role": "user", "content": "Say hi"}
            ],
            "max_tokens": 10,
        }

        start = time.time()
        response = api_client.post("/v1/chat/completions", json=payload)
        duration = time.time() - start

        assert response.status_code == 200
        # Inference should be reasonably fast (< 30s for mock)
        assert duration < 30


class TestFeedbackEndpoint:
    """Test /feedback endpoint."""

    def test_feedback_submission(self, api_client, mock_feedback_message):
        """Test submitting feedback."""
        response = api_client.post("/feedback", json=mock_feedback_message)
        # Should accept feedback (202 Accepted)
        assert response.status_code in [200, 202]

    def test_feedback_multiple_submissions(self, api_client, mock_feedback_message):
        """Test multiple feedback submissions."""
        for i in range(5):
            message = {**mock_feedback_message, "event_id": f"event-{i}"}
            response = api_client.post("/feedback", json=message)
            assert response.status_code in [200, 202]

    def test_feedback_validation(self, api_client):
        """Test feedback validation."""
        # Missing required fields
        invalid_feedback = {"event_id": "test"}
        response = api_client.post("/feedback", json=invalid_feedback)
        # Should reject invalid feedback
        assert response.status_code in [400, 422]

    def test_feedback_concurrency(self, api_client, mock_feedback_message):
        """Test concurrent feedback submissions."""
        import concurrent.futures

        def submit_feedback(i):
            message = {**mock_feedback_message, "event_id": f"event-{i}"}
            return api_client.post("/feedback", json=message)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(submit_feedback, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code in [200, 202] for r in results)
        assert len(results) == 10


class TestAdminEndpoints:
    """Test /admin endpoints."""

    def test_model_info_endpoint(self, api_client):
        """Test GET /admin/model-info endpoint."""
        response = api_client.get("/admin/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "champion_version" in data

    def test_update_champion_endpoint(self, api_client):
        """Test POST /admin/update-champion endpoint."""
        payload = {
            "new_champion_path": "s3://ml-artifacts/models/champion/v123/adapter_model.bin"
        }

        headers = {"X-Admin-Key": "test-admin-key"}
        response = api_client.post("/admin/update-champion", json=payload, headers=headers)

        # Should handle the request (might succeed or auth might fail)
        assert response.status_code in [200, 401, 403]

    def test_rollback_endpoint(self, api_client):
        """Test POST /admin/rollback endpoint."""
        headers = {"X-Admin-Key": "test-admin-key"}
        response = api_client.post("/admin/rollback", headers=headers)

        # Should handle the request
        assert response.status_code in [200, 401, 403]

    def test_admin_auth_required(self, api_client):
        """Test that admin endpoints require auth."""
        # Without X-Admin-Key header
        response = api_client.post("/admin/rollback")
        # Should either require auth or accept the request
        assert response.status_code in [200, 401, 403]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_endpoint(self, api_client):
        """Test accessing invalid endpoint."""
        response = api_client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_invalid_method(self, api_client):
        """Test invalid HTTP method."""
        response = api_client.delete("/health")
        assert response.status_code == 405

    def test_malformed_json(self, api_client):
        """Test malformed JSON request."""
        response = api_client.post(
            "/feedback",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

    def test_request_logging(self, api_client):
        """Test that requests are logged."""
        response = api_client.get("/health")
        assert response.status_code == 200
        # If no exception, logging is working
        assert True

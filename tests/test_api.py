"""Exercise real HTTP over an ephemeral loopback port; no extra test dependency."""
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import uvicorn
from api.main import app
from api.model_loader import ModelUnavailable


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.socket = socket.socket()
        cls.socket.bind(("127.0.0.1", 0))
        cls.base = f"http://127.0.0.1:{cls.socket.getsockname()[1]}"
        cls.server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={"sockets": [cls.socket]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 10
        while not cls.server.started and time.monotonic() < deadline:
            time.sleep(.02)
        if not cls.server.started:
            raise RuntimeError("API test server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(10)
        cls.socket.close()

    def request(self, path, payload=None, headers=None, method=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + path, data=body, headers=headers or {"Content-Type": "application/json"}, method=method)
        try:
            response = urlopen(request, timeout=60)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, json.loads(response.read()), response.headers

    def test_root_health_models(self):
        for path in ("/", "/health", "/models"):
            status, data, _ = self.request(path)
            self.assertEqual(status, 200)
        self.assertEqual(len(data["models"]), 4)

    def test_translate_defaults_to_scratch_beam(self):
        status, data, _ = self.request("/translate", {"text": "I am happy."})
        self.assertEqual(status, 200)
        self.assertEqual((data["model"], data["decoding"]), ("transformer", "beam"))

    def test_attention(self):
        status, data, _ = self.request("/attention", {"text": "How are you?"})
        self.assertEqual(status, 200)
        self.assertEqual(len(data["attention_matrix"]), len(data["target_tokens"]))

    def test_compare_core(self):
        status, data, _ = self.request("/compare", {"text": "India is a diverse country."})
        self.assertEqual(status, 200)
        self.assertEqual([x["model"] for x in data["results"]], ["gru", "attention", "transformer"])
        self.assertEqual(data["errors"], {})

    def test_empty_invalid_and_long(self):
        for payload in ({"text": " "}, {"text": "I", "model": "wrong"}, {"text": "I", "decoding": "wrong"}, {"text": "I" * 20001}):
            self.assertEqual(self.request("/translate", payload)[0], 422)
        status, data, _ = self.request("/translate", {"text": "I am happy. " * 80, "model": "gru"})
        self.assertEqual(status, 200)
        self.assertIn("truncated", data["warning"])

    def test_nllb_unavailable_preserves_core_comparison(self):
        with patch("api.translation_service.translate_nllb", side_effect=ModelUnavailable("NLLB unavailable")):
            with self.assertLogs("api.main", level="WARNING"):
                status, data, _ = self.request("/compare", {"text": "I am happy.", "include_nllb": True})
            self.assertEqual(status, 200)
            self.assertEqual(len(data["results"]), 3)
            self.assertIn("nllb", data["errors"])

    def test_unavailable_model_returns_503(self):
        with patch("api.translation_service.get_model", side_effect=ModelUnavailable("Missing checkpoint")), self.assertLogs("api.main", level="WARNING"):
            status, data, _ = self.request("/translate", {"text": "I am happy."})
        self.assertEqual(status, 503)
        self.assertIn("Missing checkpoint", data["detail"])

    def test_unexpected_inference_error_returns_500_without_stopping_server(self):
        with patch("api.translation_service.decode", side_effect=RuntimeError("diagnostic error")), self.assertLogs("api.main", level="ERROR"):
            self.assertEqual(self.request("/translate", {"text": "I am happy."})[0], 500)
        self.assertEqual(self.request("/health")[0], 200)

    def test_cors_local_frontend(self):
        status, _, headers = self.request("/health", headers={"Origin": "http://127.0.0.1:5500"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://127.0.0.1:5500")

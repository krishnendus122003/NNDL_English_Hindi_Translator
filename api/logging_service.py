import json
import logging
from datetime import datetime, timezone
from threading import Lock
from .config import LOG_PATH

_lock = Lock()


def log_prediction(result):
    record = {key: result[key] for key in (
        "input_text", "model", "decoding", "translation", "latency_ms", "warning")}
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        with _lock:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).exception("Prediction logging failed; translation remains available")

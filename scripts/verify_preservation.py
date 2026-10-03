"""Read-only SHA-256/size/mtime verification of every original workspace file."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "docs/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"Missing: {relative}")
            continue
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"] or path.stat().st_mtime_ns != expected["mtime_ns"]:
            failures.append(f"Changed: {relative}")
    print(f"Original files checked: {len(manifest)}; unchanged: {len(manifest) - len(failures)}; failures: {len(failures)}")
    for failure in failures:
        print(failure)
    return len(failures)


if __name__ == "__main__":
    raise SystemExit(1 if verify() else 0)

"""
IngestionManifest -- tracks per-chunk completion status for resumable,
idempotent ingestion.

Design (per docs/phase2-ingestion-design.md section 12):
  - One manifest file per source (JSON on disk).
  - A chunk is only marked "complete" AFTER its output file is successfully
    written -- never before the underlying API call/write succeeds.
  - On restart, chunks already marked complete are skipped by the caller
    (see ingestion/firms/ingest.py); this module only tracks state, it does
    not decide skip logic itself.
  - Writes are made atomic via write-to-temp-then-rename, so a crash mid-save
    cannot corrupt the manifest file itself.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class IngestionManifest:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def _save(self):
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp_path.replace(self.path)  # atomic on POSIX filesystems

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def status(self, chunk_id):
        entry = self._data.get(chunk_id)
        return entry["status"] if entry else "pending"

    def is_complete(self, chunk_id):
        return self.status(chunk_id) == "complete"

    def mark_in_progress(self, chunk_id):
        with self._lock:
            self._data[chunk_id] = {
                "status": "in_progress",
                "updated_at": self._now(),
            }
            self._save()

    def mark_complete(self, chunk_id, output_path, row_count, extra=None):
        """extra: optional dict of source-specific metadata to attach to the
        manifest entry (e.g. POWER's per-column missingness percentages).
        Backward-compatible -- existing callers that don't pass extra are
        unaffected."""
        entry = {
            "status": "complete",
            "output_path": str(output_path),
            "row_count": row_count,
            "updated_at": self._now(),
        }
        if extra:
            entry.update(extra)
        with self._lock:
            self._data[chunk_id] = entry
            self._save()

    def mark_failed(self, chunk_id, error):
        with self._lock:
            self._data[chunk_id] = {
                "status": "failed",
                "error": str(error),
                "updated_at": self._now(),
            }
            self._save()

    def summary(self):
        counts = {}
        for entry in self._data.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        return counts

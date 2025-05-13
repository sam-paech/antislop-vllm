# utils/tdpo_pairs_helper.py
"""
Thread-safe writer for 'tokenwise-dpo completion chosen / rejected' training pairs.

All generation workers build an in-memory list of examples and pass it to
`TDPOPairWriter.add_samples()` once the prompt finishes.  The writer keeps
a map   context_with_chat_template → sample   so that later replacements
overwrite earlier “bad” pairs automatically.

Flush once, at the end of the batch run.
"""
from __future__ import annotations

import json, threading
from pathlib import Path
from typing import Dict, List, Any


class TDPOPairWriter:
    def __init__(self, outfile: Path) -> None:
        self._outfile = outfile
        self._lock    = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}   # key → sample dict

    # .................................................................. #
    #  Public API                                                        #
    # .................................................................. #
    def add_samples(self, samples: List[Dict[str, Any]]) -> None:
        """
        Merge a worker’s list of samples into the global store.
        Later samples with the same context key *replace* earlier ones.
        """
        if not samples:
            return
        with self._lock:
            for s in samples:
                key = s["context_with_chat_template"]
                self._store[key] = s

    def flush(self) -> None:
        """Write all kept samples to *outfile* as JSONL."""
        with self._lock:
            if not self._store:
                return
            self._outfile.parent.mkdir(parents=True, exist_ok=True)
            with self._outfile.open("w", encoding="utf-8") as fh:
                for rec in self._store.values():
                    json.dump(rec, fh, ensure_ascii=False)
                    fh.write("\n")

# utils/ftpo_pairs_helper.py
"""
Very small, thread-safe writer for token-wise-DPO training pairs.

Every call to `add_samples()` simply APPENDS the supplied records to
`outfile` (one JSON object per line).  No deduplication, no rewriting,
no temporary files.
"""
import json
import threading
from pathlib import Path
from typing  import List, Dict, Any


class ftpoPairWriter:
    def __init__(self, outfile: Path) -> None:
        self._outfile = outfile
        self._lock    = threading.Lock()
        # Ensure the directory exists up-front so `add_samples` never has to
        self._outfile.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def add_samples(self, samples: List[Dict[str, Any]]) -> None:
        """
        Append each sample dict as a JSON line.

        If *samples* is empty the call is a no-op.
        """
        if not samples:
            return

        with self._lock, self._outfile.open("a", encoding="utf-8") as fh:
            for rec in samples:
                json.dump(rec, fh, ensure_ascii=False)
                fh.write("\n")

    def flush(self) -> None:
        """
        No-op (retained so callers don’t need changing).
        All writes happen immediately in `add_samples()`.
        """
        pass

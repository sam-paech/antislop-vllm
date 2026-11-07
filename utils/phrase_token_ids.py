#!/usr/bin/env python3
"""
Convert a flat JSON list of slop phrases into an AntiSlop banlist
[ ["phrase", first_token_id], ... ] using a given tokenizer.

Usage:
  python phrase_token_ids.py \
      --phrases  /path/to/slop_phrases_flat.json \
      --tokenizer /path/to/tokenizer.json \
      --out      /path/to/finished.json \
      --report   /path/to/finished.report.txt
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("tokenizers is required: pip install tokenizers")

QUOTE_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def load_phrases_with_repair(path: Path):
    """Load a JSON list of strings; on failure, patch missing commas between
    two adjacent ' "str"' lines and retry."""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw), []
    except json.JSONDecodeError as e:
        print(f"[repair] initial parse failed: {e}", file=sys.stderr)
        lines = raw.splitlines()
        fixed, repairs = [], []
        for i, line in enumerate(lines):
            s = line.strip()
            if i > 0:
                ps = fixed[-1].strip()
                # previous line is a bare "string" with no trailing comma and
                # this line begins with a quoted string -> missing comma
                if QUOTE_RE.fullmatch(ps) and QUOTE_RE.match(s):
                    fixed[-1] = fixed[-1] + ","
                    repairs.append((i + 1, ps, s))
            fixed.append(line)
        repaired = "\n".join(fixed)
        try:
            return json.loads(repaired), repairs
        except json.JSONDecodeError as e2:
            pos = e2.pos
            print(f"[repair] still failing: {e2}\n...{repaired[max(0, pos-300):pos+300]}...",
                  file=sys.stderr)
            raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases",   required=True, help="flat JSON list of phrases")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--out",       required=True, help="output banlist path")
    ap.add_argument("--report",    default=None,  help="optional report path")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    print(f"[tok] vocab size = {vocab_size}", file=sys.stderr)

    phrases, repairs = load_phrases_with_repair(Path(args.phrases))
    if not isinstance(phrases, list):
        raise SystemExit("root of phrases file is not a list")
    print(f"[load] {len(phrases)} raw phrases, {len(repairs)} comma repairs", file=sys.stderr)
    for ln, a, b in repairs:
        print(f"   after line {ln}: {a}  /  {b}", file=sys.stderr)

    # normalize: lowercase, dedupe (keep first occurrence + order), drop empty
    seen, ordered, dropped = set(), [], []
    for p in phrases:
        if not isinstance(p, str):
            dropped.append((repr(p), "not a string")); continue
        lp = p.lower()
        if lp == "" or lp.strip() == "":
            dropped.append((repr(p), "empty")); continue
        if lp in seen:
            dropped.append((repr(p), f"duplicate of {lp!r}")); continue
        seen.add(lp); ordered.append(lp)
    print(f"[dedupe] {len(ordered)} kept, {len(dropped)} dropped", file=sys.stderr)

    out, full_seqs, problems, single = [], {}, [], 0
    for p in ordered:
        enc = tok.encode(p, add_special_tokens=False)
        ids = enc.ids
        dec = tok.decode(ids, skip_special_tokens=True)
        if not ids:
            problems.append((p, "encodes to zero tokens")); continue
        if max(ids) >= vocab_size:
            problems.append((p, f"id {max(ids)} outside vocab {vocab_size}")); continue
        if dec != p:
            problems.append((p, f"roundtrip mismatch: decode={dec!r}")); continue
        full_seqs[p] = ids
        out.append([p, ids[0]])          # <-- FIRST token id
        if len(ids) == 1:
            single += 1

    Path(args.out).write_text(
        json.dumps(out, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[out] wrote {args.out}: {len(out)} entries "
          f"({single} single-token, {len(out)-single} multi-token)", file=sys.stderr)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("slop phrase -> token id report\n")
            fh.write("==============================\n\n")
            fh.write(f"phrases file : {args.phrases}\n")
            fh.write(f"tokenizer    : {args.tokenizer} (vocab {vocab_size})\n")
            fh.write("id stored    : FIRST token of lowercase phrase "
                     "(encode(..., add_special_tokens=False)[0])\n")
            fh.write(f"raw          : {len(phrases)}\n")
            fh.write(f"repairs      : {len(repairs)}\n")
            fh.write(f"dropped      : {len(dropped)}\n")
            fh.write(f"kept         : {len(out)}\n\n")
            fh.write("--- dropped ---\n")
            for d, why in dropped:
                fh.write(f"  {d}: {why}\n")
            fh.write("\n--- excluded (problems) ---\n")
            if problems:
                for p, why in problems:
                    fh.write(f"  {p!r}: {why}\n")
            else:
                fh.write("  none\n")
            fh.write("\n--- full token id sequences (reference) ---\n")
            for p, ids in full_seqs.items():
                fh.write(f"  {p!r} -> {ids}\n")
        print(f"[report] wrote {args.report}", file=sys.stderr)

    if problems:
        print("[warn] some phrases excluded - see report", file=sys.stderr)


if __name__ == "__main__":
    main()

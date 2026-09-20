#!/usr/bin/env python3
"""Runaway-response detector for JSONL chat-completion logs.

Scans a JSONL file where each line is a JSON object with fields:
  id, prompt_tokens, completion_tokens, wall_s, text

Flags a response as runaway when ANY of:
  - completion_tokens > 1500
  - wall_s > 60
  - the text contains a run of >= 20 identical whitespace-separated tokens

These thresholds flag CANDIDATE runaways: a legitimately long or slow response can
also match, so every flagged line should be reviewed before acting on it. A missing
completion_tokens / wall_s / text field is not treated as a runaway (it is reported
as a malformed record), and a text value that is not a string is reported rather than
crashing the scan.

Prints a one-line summary and exits with code 2 if any response was flagged,
0 only when the file had >= 1 valid record and none were flagged, and 1 on a
file-level error (file not found / IO error / encoding error) or when the file
had zero valid records (empty file or every line failed to parse), so a clean
scan is never confused with an empty or unreadable input.

Pure standard library. Run `--selftest` for a tiny built-in self-test.
"""
import argparse
import json
import sys
from pathlib import Path

# Detection thresholds (verbatim from the cookbook's runaway symptom).
MAX_COMPLETION_TOKENS = 1500
MAX_WALL_S = 60
MIN_REPEAT_RUN = 20


def has_long_repeat(text, min_run=MIN_REPEAT_RUN):
    """True if `text` contains >= `min_run` identical whitespace-separated tokens
    in a row."""
    if not isinstance(text, str) or not text:
        return False
    tokens = text.split()
    if len(tokens) < min_run:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1
    return False


def classify(record, warn=sys.stderr.write):
    """Return a list of human-readable reasons the record is a runaway, or [].

    A record missing the required fields, or with a non-string `text`, is not a
    runaway; the malformation is reported via `warn` instead."""
    reasons = []
    ct = record.get("completion_tokens")
    if ct is not None and not isinstance(ct, (int, float)):
        warn("warn: record has non-numeric completion_tokens={!r}\n".format(ct))
        ct = None
    if isinstance(ct, (int, float)) and ct > MAX_COMPLETION_TOKENS:
        reasons.append("completion_tokens={} > {}".format(ct, MAX_COMPLETION_TOKENS))
    wall = record.get("wall_s")
    if wall is not None and not isinstance(wall, (int, float)):
        warn("warn: record has non-numeric wall_s={!r}\n".format(wall))
        wall = None
    if isinstance(wall, (int, float)) and wall > MAX_WALL_S:
        reasons.append("wall_s={} > {}".format(wall, MAX_WALL_S))
    text = record.get("text", "")
    if text is not None and not isinstance(text, str):
        warn("warn: record has non-string text={!r}; skipping repeat check\n".format(text))
    elif has_long_repeat(text):
        reasons.append(">= {} identical tokens in a row".format(MIN_REPEAT_RUN))
    return reasons


class ScanStats(object):
    """Counters that let main() tell a clean file apart from an empty/broken one."""

    def __init__(self):
        self.valid = 0
        self.parse_errors = 0
        self.non_object = 0
        self.flagged = 0


def scan(path, stats=None, warn=sys.stderr.write):
    """Yield (id, reasons) tuples for every flagged record in the JSONL file.

    Records per-line JSON errors are reported on stderr and skipped (a malformed
    line is not a runaway). File-level IO / encoding errors propagate to the
    caller as exceptions; main() turns them into a fatal exit code."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                if stats is not None:
                    stats.parse_errors += 1
                warn("warn: line {}: not JSON: {}\n".format(lineno, exc))
                continue
            if not isinstance(rec, dict):
                if stats is not None:
                    stats.non_object += 1
                warn("warn: line {}: not a JSON object\n".format(lineno))
                continue
            if stats is not None:
                stats.valid += 1
            reasons = classify(rec, warn=warn)
            if reasons:
                if stats is not None:
                    stats.flagged += 1
                yield rec.get("id", "line:{}".format(lineno)), reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="JSONL file of chat completions to scan")
    args = ap.parse_args(argv)

    path = Path(args.file)
    if not path.is_file():
        sys.stderr.write("error: not a file: {}\n".format(path))
        return 1

    stats = ScanStats()
    try:
        flagged = list(scan(path, stats=stats))
    except OSError as exc:
        sys.stderr.write("error: could not read {}: {}\n".format(path, exc))
        return 1
    except UnicodeDecodeError as exc:
        sys.stderr.write("error: not utf-8: {}: {}\n".format(path, exc))
        return 1

    if flagged:
        for rid, reasons in flagged:
            print("FLAGGED {}: {}".format(rid, "; ".join(reasons)))
        print("summary: {} runaway response(s) flagged in {} "
              "({} valid records, {} parse errors, {} non-object)".format(
                  stats.flagged, path, stats.valid, stats.parse_errors, stats.non_object))
        return 2
    if stats.valid == 0:
        sys.stderr.write("error: no valid records scanned in {} "
                         "({} parse errors, {} non-object)\n".format(
                             path, stats.parse_errors, stats.non_object))
        return 1
    print("summary: 0 runaway responses in {} "
          "({} valid records, {} parse errors, {} non-object)".format(
              path, stats.valid, stats.parse_errors, stats.non_object))
    return 0


# --- tiny self-test ---------------------------------------------------------
def _selftest():
    """Exercise the three detection rules plus a clean record. Returns 0 on pass."""
    cases = [
        # (record, expect_flagged)
        ({"id": "ok", "prompt_tokens": 10, "completion_tokens": 100, "wall_s": 5,
          "text": "the answer is forty-two"}, False),
        ({"id": "too-long", "prompt_tokens": 10, "completion_tokens": 2000, "wall_s": 5,
          "text": "short"}, True),
        ({"id": "too-slow", "prompt_tokens": 10, "completion_tokens": 100, "wall_s": 61,
          "text": "short"}, True),
        ({"id": "repeat", "prompt_tokens": 10, "completion_tokens": 100, "wall_s": 5,
          "text": "the " * 25}, True),
        ({"id": "almost-repeat", "prompt_tokens": 10, "completion_tokens": 100, "wall_s": 5,
          "text": "the " * 19 + "end"}, False),
        ({"id": "repeat-across-punct", "prompt_tokens": 10, "completion_tokens": 100,
          "wall_s": 5, "text": " ".join(["stop"] * 20)}, True),
        # non-string text must not crash (malformed, not a runaway)
        ({"id": "bad-text", "prompt_tokens": 10, "completion_tokens": 100, "wall_s": 5,
          "text": ["x"]}, False),
    ]
    failures = 0
    import io
    for rec, expect in cases:
        sink = io.StringIO()
        got = bool(classify(rec, warn=sink.write))
        if got != expect:
            failures += 1
            print("FAIL {}: expected {} got {}".format(rec["id"], expect, got))

    # exercise the file path via scan() using a temp JSONL in-memory list.
    import tempfile, os
    lines = [json.dumps(r) for r, _ in cases]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write("\n".join(lines) + "\n")
        tmp = tf.name
    try:
        stats = ScanStats()
        flagged = list(scan(tmp, stats=stats))
        flagged_ids = {rid for rid, _ in flagged}
        expected_ids = {"too-long", "too-slow", "repeat", "repeat-across-punct"}
        if flagged_ids != expected_ids:
            failures += 1
            print("FAIL scan: expected {} got {}".format(sorted(expected_ids), sorted(flagged_ids)))
        if stats.valid != len(cases) or stats.parse_errors != 0 or stats.non_object != 0:
            failures += 1
            print("FAIL stats: valid={} parse={} nonobj={}".format(
                stats.valid, stats.parse_errors, stats.non_object))

        # an all-malformed file must yield zero valid records (not a clean scan)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf2:
            tf2.write("not json\n[1, 2]\n")
            bad = tf2.name
        bad_stats = ScanStats()
        bad_flagged = list(scan(bad, stats=bad_stats))
        if bad_flagged or bad_stats.valid != 0 or bad_stats.parse_errors != 1 or bad_stats.non_object != 1:
            failures += 1
            print("FAIL empty: flagged={} valid={} parse={} nonobj={}".format(
                len(bad_flagged), bad_stats.valid, bad_stats.parse_errors, bad_stats.non_object))
        os.unlink(bad)
    finally:
        os.unlink(tmp)

    if failures:
        print("selftest: {} FAIL".format(failures))
        return 1
    print("selftest: OK ({} cases)".format(len(cases)))
    return 0


if __name__ == "__main__":
    # --selftest runs the built-in self-test and ignores any file argument.
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())

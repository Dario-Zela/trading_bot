"""Phase 7 — Archive old editions.

Walks `docs/news/YYYY-MM-DD/` and `docs/macro/YYYY-W##/` directories
(plus the legacy `docs/news/YYYY-MM-DD.html` flat files) and bundles
anything older than 90 days into a compressed tarball at
`state/archive/{news,macro}-YYYY-MM.tar.gz`, removing the originals
from `docs/` so the working tree stays slim.

A manifest at `state/archive/manifest.json` records what's been
trimmed so the news archive page can still link to the compressed
form (raw GitHub blob URL).

Idempotent: running twice is a no-op the second time.

Usage
-----
    python scripts/archive_old_editions.py
    python scripts/archive_old_editions.py --keep-days 60
    python scripts/archive_old_editions.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tarfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

log = logging.getLogger("archive_old_editions")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOCS_ROOT = _REPO_ROOT / "docs"
_STATE_ROOT = _REPO_ROOT / "state"
_ARCHIVE_ROOT = _STATE_ROOT / "archive"
_MANIFEST_PATH = _ARCHIVE_ROOT / "manifest.json"

_DEFAULT_KEEP_DAYS = 90
# Public GitHub blob URL prefix for the archived tarballs (so the
# news index can deep-link to them even after they're trimmed from docs).
_BLOB_URL_PREFIX = "https://github.com/Dario-Zela/trading_bot/raw/main/state/archive"


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        return {"news": [], "macro": [], "last_run": None}
    try:
        return json.loads(_MANIFEST_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("Manifest corrupt — starting fresh")
        return {"news": [], "macro": [], "last_run": None}


def _save_manifest(m: dict, *, dry_run: bool) -> None:
    if dry_run:
        log.info("[dry-run] would write manifest with %d news bundles, %d macro bundles",
                 len(m.get("news", [])), len(m.get("macro", [])))
        return
    _ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(m, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _news_candidates(cutoff: date) -> list[tuple[Path, str]]:
    """Return list of (path, date_iso) pairs for news entries older
    than `cutoff`. Covers both dir-form and legacy flat-file form."""
    out: list[tuple[Path, str]] = []
    news_dir = _DOCS_ROOT / "news"
    if not news_dir.exists():
        return out

    for child in news_dir.iterdir():
        # Directory-form editions: docs/news/YYYY-MM-DD/
        if child.is_dir():
            try:
                d = datetime.strptime(child.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < cutoff:
                out.append((child, d.isoformat()))
        # Legacy flat-file editions: docs/news/YYYY-MM-DD.html
        elif child.is_file() and child.suffix == ".html":
            stem = child.stem
            if stem == "index":
                continue
            try:
                d = datetime.strptime(stem, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < cutoff:
                out.append((child, d.isoformat()))
    return out


def _macro_candidates(cutoff: date) -> list[tuple[Path, str]]:
    """Return list of (path, week_iso) for macro entries older than
    `cutoff`. Macro is dir-form only in Phase 3; legacy was flat."""
    out: list[tuple[Path, str]] = []
    macro_dir = _DOCS_ROOT / "macro"
    if not macro_dir.exists():
        return out

    for child in macro_dir.iterdir():
        if child.is_dir():
            # YYYY-W## → use Monday of that ISO week as the comparison date
            wd = _iso_week_to_date(child.name)
            if wd is None:
                continue
            if wd < cutoff:
                out.append((child, child.name))
        elif child.is_file() and child.suffix == ".html":
            stem = child.stem
            if stem == "index":
                continue
            wd = _iso_week_to_date(stem)
            if wd is None:
                continue
            if wd < cutoff:
                out.append((child, stem))
    return out


def _iso_week_to_date(week_id: str) -> date | None:
    """Convert 'YYYY-WNN' → Monday of that ISO week."""
    try:
        year_str, week_str = week_id.split("-W")
        return date.fromisocalendar(int(year_str), int(week_str), 1)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Bundling
# ---------------------------------------------------------------------------

def _bundle_news(entries: list[tuple[Path, str]], *, dry_run: bool) -> list[dict]:
    """Bundle news entries grouped by YYYY-MM. Returns manifest records."""
    by_month: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, iso in entries:
        by_month[iso[:7]].append((path, iso))   # iso = "YYYY-MM-DD"

    records: list[dict] = []
    for ym, group in sorted(by_month.items()):
        tarname = f"news-{ym}.tar.gz"
        record = _make_bundle(group, tarname, dry_run=dry_run, kind="news")
        if record:
            record["entries"] = sorted(iso for _, iso in group)
            records.append(record)
    return records


def _bundle_macro(entries: list[tuple[Path, str]], *, dry_run: bool) -> list[dict]:
    """Bundle macro entries grouped by year. Returns manifest records."""
    by_year: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, week_id in entries:
        year = week_id.split("-W")[0] if "-W" in week_id else week_id[:4]
        by_year[year].append((path, week_id))

    records: list[dict] = []
    for year, group in sorted(by_year.items()):
        tarname = f"macro-{year}.tar.gz"
        record = _make_bundle(group, tarname, dry_run=dry_run, kind="macro")
        if record:
            record["entries"] = sorted(wid for _, wid in group)
            records.append(record)
    return records


def _make_bundle(group: list[tuple[Path, str]], tarname: str, *, dry_run: bool, kind: str) -> dict | None:
    """Compress the group's paths into a tarball at state/archive/<tarname>
    and remove the originals. Returns a manifest record or None if no-op."""
    if not group:
        return None
    _ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    tar_path = _ARCHIVE_ROOT / tarname

    if dry_run:
        log.info("[dry-run] would archive %d %s entries → %s",
                 len(group), kind, tar_path.name)
        return {
            "filename": tarname,
            "url": f"{_BLOB_URL_PREFIX}/{tarname}",
            "kind": kind,
            "n_entries": len(group),
            "bundled_at": None,
        }

    # Append-or-replace: if the tarball exists, we extract + re-tar to
    # merge new entries. Simple and rare enough that we don't optimise.
    existing_paths: list[Path] = []
    if tar_path.exists():
        with tarfile.open(tar_path, "r:gz") as tf:
            members = tf.getnames()
        log.info("%s already exists with %d members — merging", tar_path.name, len(members))
        # Extract to a temp dir, then re-tar with new entries added
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(tmp_path, filter="data")  # tarfile filter requires py>=3.12 — fine on 3.11 fallback below
            # Move existing members alongside our new ones
            for name in members:
                p = tmp_path / name
                if p.exists():
                    existing_paths.append(p)
            tar_path.unlink()
            with tarfile.open(tar_path, "w:gz") as tf:
                for p in existing_paths:
                    tf.add(p, arcname=p.name)
                for path, _ in group:
                    tf.add(path, arcname=path.name)
    else:
        with tarfile.open(tar_path, "w:gz") as tf:
            for path, _ in group:
                tf.add(path, arcname=path.name)

    # Remove originals
    for path, _ in group:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    log.info("Archived %d %s entries → %s (%.1f KB)",
             len(group), kind, tar_path.name, tar_path.stat().st_size / 1024)

    return {
        "filename": tarname,
        "url": f"{_BLOB_URL_PREFIX}/{tarname}",
        "kind": kind,
        "n_entries": len(group),
        "bundled_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trim old news/macro editions into compressed archives.")
    parser.add_argument("--keep-days", type=int, default=_DEFAULT_KEEP_DAYS,
                        help=f"Keep editions younger than this many days (default {_DEFAULT_KEEP_DAYS}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be archived without writing anything.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    cutoff = date.today() - timedelta(days=args.keep_days)
    log.info("Cutoff: archiving anything older than %s (%d days)", cutoff.isoformat(), args.keep_days)

    news_entries = _news_candidates(cutoff)
    macro_entries = _macro_candidates(cutoff)
    log.info("Candidates: %d news, %d macro", len(news_entries), len(macro_entries))

    if not news_entries and not macro_entries:
        log.info("Nothing to archive — working tree is already clean.")
        return 0

    manifest = _load_manifest()

    new_news_records = _bundle_news(news_entries, dry_run=args.dry_run)
    new_macro_records = _bundle_macro(macro_entries, dry_run=args.dry_run)

    # Merge manifest: dedupe by filename, prefer the freshest record
    def _merge(existing: list[dict], new: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {r["filename"]: r for r in existing if isinstance(r, dict) and "filename" in r}
        for r in new:
            existing_record = merged.get(r["filename"])
            if existing_record:
                # Merge entries lists
                merged_entries = sorted(set((existing_record.get("entries") or []) + (r.get("entries") or [])))
                r = {**r, "entries": merged_entries, "n_entries": len(merged_entries)}
            merged[r["filename"]] = r
        return sorted(merged.values(), key=lambda r: r["filename"])

    manifest["news"] = _merge(manifest.get("news", []), new_news_records)
    manifest["macro"] = _merge(manifest.get("macro", []), new_macro_records)
    manifest["last_run"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    _save_manifest(manifest, dry_run=args.dry_run)
    log.info("Done. Wrote %d news bundle(s), %d macro bundle(s).",
             len(new_news_records), len(new_macro_records))
    return 0


if __name__ == "__main__":
    sys.exit(main())

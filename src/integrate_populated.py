#!/usr/bin/env python3
"""Build integrated dataset from structural features and enriched metadata.

This utility waits briefly for an upstream metadata file, merges it with
structural features on ``uniprot_id`` using existing integration utilities,
writes the integrated dataset CSV, and emits a validation log containing join
retention and missingness metrics.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.metadata import integrate_features

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = Path("data/processed/features.csv")
DEFAULT_METADATA_PATH = Path("data/processed/metadata_populated.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/integrated_populated.csv")
DEFAULT_LOG_PATH = Path("logs/integrated_populated_validation.log")


def configure_logging() -> None:
    """Configure console logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def wait_for_file(path: Path, attempts: int, sleep_seconds: int) -> bool:
    """Wait for *path* to exist, checking every *sleep_seconds* up to *attempts*."""
    for attempt in range(1, attempts + 1):
        if path.exists():
            logger.info("Found metadata file on attempt %d: %s", attempt, path)
            return True
        logger.info(
            "Attempt %d/%d: metadata file not found at %s; waiting %ds",
            attempt,
            attempts,
            path,
            sleep_seconds,
        )
        time.sleep(sleep_seconds)
    return False


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read CSV file at *path* as a list of dictionaries."""
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def csv_headers(path: Path) -> List[str]:
    """Return CSV headers from *path*."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    """Write *rows* to *path* with fieldnames inferred from first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty integrated dataset.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _missing_count(values: Iterable[str]) -> int:
    """Count missing-like values (empty/None/NA markers)."""
    missing_tokens = {"", "na", "n/a", "none", "null", "nan"}
    count = 0
    for value in values:
        if value is None:
            count += 1
            continue
        if str(value).strip().lower() in missing_tokens:
            count += 1
    return count


def missingness_by_field(
    rows: Sequence[Dict[str, str]],
    fields: Sequence[str],
) -> List[Tuple[str, int, float]]:
    """Return missingness metrics as ``(field, missing_count, missing_pct)``."""
    total = len(rows)
    metrics: List[Tuple[str, int, float]] = []
    for field in fields:
        missing_count = _missing_count(row.get(field) for row in rows)
        missing_pct = (missing_count / total * 100.0) if total else 0.0
        metrics.append((field, missing_count, missing_pct))
    return metrics


def _format_missingness_block(
    title: str,
    rows: Sequence[Dict[str, str]],
    fields: Sequence[str],
) -> List[str]:
    lines = [f"{title} missingness:"]
    if not fields:
        lines.append("  - no fields available")
        return lines
    for field, missing_count, missing_pct in missingness_by_field(rows, fields):
        lines.append(f"  - {field}: {missing_count}/{len(rows)} ({missing_pct:.2f}%)")
    return lines


def _join_retention(integrated_count: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return integrated_count / denominator * 100.0


def build_validation_log(
    *,
    features_rows: Sequence[Dict[str, str]],
    metadata_rows: Sequence[Dict[str, str]],
    integrated_rows: Sequence[Dict[str, str]],
    feature_fields: Sequence[str],
    env_fields: Sequence[str],
) -> str:
    """Build a human-readable validation report."""
    integrated_count = len(integrated_rows)
    features_count = len(features_rows)
    metadata_count = len(metadata_rows)

    feature_ids = {row.get("uniprot_id", "") for row in features_rows}
    metadata_ids = {row.get("uniprot_id", "") for row in metadata_rows}
    integrated_ids = {row.get("uniprot_id", "") for row in integrated_rows}

    lines = [
        "integrated_populated validation report",
        f"generated_utc: {datetime.now(timezone.utc).isoformat()}",
        "",
        "row counts:",
        f"  - features_rows: {features_count}",
        f"  - metadata_rows: {metadata_count}",
        f"  - integrated_rows: {integrated_count}",
        "",
        "join retention:",
        (
            f"  - integrated_vs_features: {integrated_count}/{features_count} "
            f"({_join_retention(integrated_count, features_count):.2f}%)"
        ),
        (
            f"  - integrated_vs_metadata: {integrated_count}/{metadata_count} "
            f"({_join_retention(integrated_count, metadata_count):.2f}%)"
        ),
        f"  - unmatched_feature_ids: {len(feature_ids - integrated_ids)}",
        f"  - unmatched_metadata_ids: {len(metadata_ids - integrated_ids)}",
        "",
    ]
    lines.extend(_format_missingness_block("structural fields", integrated_rows, feature_fields))
    lines.append("")
    lines.extend(_format_missingness_block("env fields", integrated_rows, env_fields))
    lines.append("")
    return "\n".join(lines)


def write_log(path: Path, text: str) -> None:
    """Write text log to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(
    *,
    features_path: Path,
    metadata_path: Path,
    output_path: Path,
    log_path: Path,
    wait_attempts: int,
    wait_seconds: int,
) -> int:
    """Execute integration workflow. Returns process exit code."""
    if not features_path.exists():
        logger.error("Features file not found: %s", features_path)
        return 1

    if not wait_for_file(metadata_path, attempts=wait_attempts, sleep_seconds=wait_seconds):
        logger.error(
            "Blocker: metadata file unavailable after %d attempts: %s",
            wait_attempts,
            metadata_path,
        )
        logger.info(
            "Per policy, output files were not written or modified: %s, %s",
            output_path,
            log_path,
        )
        return 2

    features_rows = read_csv_rows(features_path)
    metadata_rows = read_csv_rows(metadata_path)
    integrated_rows = integrate_features(metadata_rows=metadata_rows, feature_rows=features_rows)

    if not integrated_rows:
        logger.error("Integration produced 0 rows; refusing to write empty output.")
        return 1

    write_csv_rows(output_path, integrated_rows)

    feature_fields = [name for name in csv_headers(features_path) if name != "uniprot_id"]
    env_fields = [name for name in csv_headers(metadata_path) if name != "uniprot_id"]
    validation_text = build_validation_log(
        features_rows=features_rows,
        metadata_rows=metadata_rows,
        integrated_rows=integrated_rows,
        feature_fields=feature_fields,
        env_fields=env_fields,
    )
    write_log(log_path, validation_text)

    logger.info("Wrote integrated dataset: %s (%d rows)", output_path, len(integrated_rows))
    logger.info("Wrote validation log: %s", log_path)
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Build integrated_populated.csv from features.csv + metadata_populated.csv",
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--wait-attempts", type=int, default=12)
    parser.add_argument("--wait-seconds", type=int, default=5)
    args = parser.parse_args()

    configure_logging()
    return run(
        features_path=args.features,
        metadata_path=args.metadata,
        output_path=args.output,
        log_path=args.validation_log,
        wait_attempts=args.wait_attempts,
        wait_seconds=args.wait_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())

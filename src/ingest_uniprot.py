"""UniProt ingestion utilities for marine metallothionein sequences.

The module supports both library use and command-line execution. The exported
CSV schema is intentionally narrow so downstream feature extraction can rely on
stable column names.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from src.uniprot_taxa import get_marine_taxon_ids


DEFAULT_OUTPUT = "data/raw/uniprot_sequences.csv"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_PROTEIN_NAME = "metallothionein"
DEFAULT_ORGANISM_IDS = tuple(get_marine_taxon_ids())
DEFAULT_MIN_LENGTH = 40
DEFAULT_MAX_LENGTH = 100
DEFAULT_MIN_CYSTEINES = 10
DEFAULT_PAGE_SIZE = 500
RE_NEXT_LINK = re.compile(r'<([^>]+)>; rel="next"')


def _configure_logging(log_path: str = "logs/data_ingestion.log") -> None:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def read_sequences_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            rows.append(row)
    return rows


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "mtsef-ingest/1.0"})
    return session


def write_sequences_csv(path: str, rows: List[Dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "uniprot_id",
        "organism",
        "sequence",
        "length",
        "cysteine_count",
        "reviewed",
    ]
    with output_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_uniprot_query(
    protein_name: str = DEFAULT_PROTEIN_NAME,
    organism_ids: Optional[Iterable[int]] = None,
    reviewed: bool = True,
) -> str:
    query_parts: List[str] = []
    if reviewed:
        query_parts.append("reviewed:true")
    if protein_name:
        query_parts.append(f'protein_name:"{protein_name}"')

    normalized_organism_ids = [str(organism_id) for organism_id in organism_ids or [] if str(organism_id)]
    if normalized_organism_ids:
        organism_filter = " OR ".join(f"organism_id:{organism_id}" for organism_id in normalized_organism_ids)
        query_parts.append(f"({organism_filter})")

    return " AND ".join(query_parts)


def validate_sequence(
    sequence: str,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_cysteines: int = DEFAULT_MIN_CYSTEINES,
) -> bool:
    if not sequence:
        return False
    length = len(sequence)
    if length < min_length or length > max_length:
        return False
    allowed_amino_acids = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    normalized_sequence = sequence.upper()
    if any(residue not in allowed_amino_acids for residue in normalized_sequence):
        return False
    return normalized_sequence.count("C") >= min_cysteines


def _get_next_link(headers: Dict[str, str]) -> Optional[str]:
    link_header = headers.get("Link")
    if not link_header:
        return None
    match = RE_NEXT_LINK.search(link_header)
    if not match:
        return None
    return match.group(1)


def _search_params(query_term: str, size: int) -> Dict[str, object]:
    return {
        "query": query_term,
        "format": "json",
        "size": size,
    }


def _http_get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, object]] = None,
    timeout: int = 30,
) -> requests.Response:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def _extract_record(entry: Dict[str, object]) -> Dict[str, str]:
    sequence_section = entry.get("sequence", {})
    organism_section = entry.get("organism", {})

    sequence = sequence_section.get("value", "") if isinstance(sequence_section, dict) else ""
    organism = organism_section.get("scientificName", "") if isinstance(organism_section, dict) else ""
    accession = entry.get("primaryAccession", "")
    reviewed = "reviewed" in str(entry.get("entryType", "")).lower()
    return {
        "uniprot_id": str(accession),
        "organism": str(organism),
        "sequence": str(sequence),
        "length": str(len(sequence)),
        "cysteine_count": str(sequence.upper().count("C")),
        "reviewed": "yes" if reviewed else "no",
    }


def fetch_uniprot_sequences(
    query_term: Optional[str] = None,
    limit: int = 50,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_cysteines: int = DEFAULT_MIN_CYSTEINES,
    protein_name: str = DEFAULT_PROTEIN_NAME,
    organism_ids: Optional[Iterable[int]] = None,
    reviewed: bool = True,
    timeout: int = 30,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen_accessions = set()
    page_size = min(max(limit, 1), DEFAULT_PAGE_SIZE)
    session = _create_session()
    url = UNIPROT_SEARCH_URL
    effective_organism_ids = tuple(organism_ids) if organism_ids is not None else DEFAULT_ORGANISM_IDS
    params = _search_params(
        query_term or build_uniprot_query(
            protein_name=protein_name,
            organism_ids=effective_organism_ids,
            reviewed=reviewed,
        ),
        page_size,
    )

    while len(rows) < limit and url:
        response = _http_get_json(session, url, params=params, timeout=timeout)
        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not results:
            break

        for entry in results:
            if not isinstance(entry, dict):
                continue
            record = _extract_record(entry)
            accession = record["uniprot_id"]
            if accession in seen_accessions:
                continue
            seen_accessions.add(accession)
            sequence = record["sequence"]
            if not validate_sequence(
                sequence,
                min_length=min_length,
                max_length=max_length,
                min_cysteines=min_cysteines,
            ):
                continue
            rows.append(record)
            if len(rows) >= limit:
                break

        url = _get_next_link(response.headers)
        params = None

    return rows[:limit]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch marine metallothionein sequences from UniProt and save them as CSV."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV file for sequences.",
    )
    parser.add_argument(
        "--query-term",
        default=None,
        help="Raw UniProt query string. If omitted, a query is built from the structured options.",
    )
    parser.add_argument(
        "--protein-name",
        default=DEFAULT_PROTEIN_NAME,
        help="Protein name to search for when query-term is not supplied.",
    )
    parser.add_argument(
        "--organism-id",
        dest="organism_ids",
        action="append",
        type=int,
        default=[],
        help="UniProt organism taxon ID to include. Repeat for multiple curated marine IDs.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of sequences to fetch.")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH, help="Minimum sequence length.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum sequence length.")
    parser.add_argument("--min-cysteines", type=int, default=DEFAULT_MIN_CYSTEINES, help="Minimum cysteine count.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--log-file",
        default="logs/data_ingestion.log",
        help="Path to the ingestion log file.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_file)

    query_term = args.query_term or build_uniprot_query(
        protein_name=args.protein_name,
        organism_ids=args.organism_ids or get_marine_taxon_ids(),
        reviewed=True,
    )

    logging.info("Starting UniProt ingestion")
    logging.info("Query: %s", query_term)
    logging.info("Requested limit: %s", args.limit)

    rows = fetch_uniprot_sequences(
        query_term=query_term,
        limit=args.limit,
        min_length=args.min_length,
        max_length=args.max_length,
        min_cysteines=args.min_cysteines,
        protein_name=args.protein_name,
        organism_ids=args.organism_ids or get_marine_taxon_ids(),
        timeout=args.timeout,
    )

    write_sequences_csv(args.output, rows)
    logging.info("Saved %s filtered sequences to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv
import json
import logging
import sys
from pathlib import Path

import yaml

from extract import extract_note
from validate import validate_extraction


# Project root is the folder above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(relative_path):
    """Convert a project-relative path into an absolute path."""
    return PROJECT_ROOT / relative_path


def load_text_file(file_path, encoding="utf-8"):
    """Read a text file."""
    with file_path.open("r", encoding=encoding) as file:
        return file.read().strip()


def load_yaml_file(file_path):
    """Read a YAML file."""
    with file_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_json_file(file_path):
    """Read a JSON file."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def setup_logging(log_file):
    """Write errors to logs/pipeline_errors.log and messages to the terminal."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def get_existing_note_ids(csv_path):
    """Return note IDs that are already in the output CSV."""
    note_ids = set()

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return note_ids

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                note_id = row.get("note_id", "").strip()
                if note_id:
                    note_ids.add(note_id)
    except (csv.Error, UnicodeDecodeError) as error:
        logging.warning("Could not read existing CSV: %s", error)

    return note_ids


def clean_csv_value(value, separator):
    """Convert lists and other values into safe text for a CSV cell."""
    if value is None:
        return ""

    if isinstance(value, list):
        return separator.join(
            str(item).strip().replace("\n", " ")
            for item in value
            if item is not None and str(item).strip()
        )

    return str(value).strip().replace("\n", " ")


def make_csv_row(extraction, csv_columns, settings, note_id):
    """Fill missing columns with defaults and prepare one CSV row."""
    output_settings = settings.get("output", {})
    list_separator = output_settings.get("multi_value_separator", ", ")
    evidence_separator = output_settings.get("evidence_separator", ", ")

    row = {}

    for column_name, default_value in csv_columns.items():
        value = extraction.get(column_name, default_value)

        if column_name.endswith("_evidence"):
            row[column_name] = clean_csv_value(value, evidence_separator)
        else:
            row[column_name] = clean_csv_value(value, list_separator)

    row["note_id"] = note_id
    return row


def append_csv_row(csv_path, row, fieldnames, encoding="utf-8"):
    """Append one row to the output CSV and write its header when needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding=encoding) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )

        if needs_header:
            writer.writeheader()

        writer.writerow(row)


def save_failed_note(failed_log_path, note_id, error):
    """Append one failed note ID and its error message to a log file."""
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)

    with failed_log_path.open("a", encoding="utf-8") as file:
        file.write(f"{note_id} | {error}\n")


def truncate_note(note_text, settings):
    input_settings = settings.get("input", {})
    max_characters = input_settings.get("max_note_characters", 12000)

    if len(note_text) <= max_characters:
        return note_text

    strategy = input_settings.get("truncation_strategy", "head")
    
    if strategy == "head_and_tail":
        head_characters = input_settings.get("head_characters", max_characters // 2)
        tail_characters = input_settings.get("tail_characters", max_characters // 2)

        return (
            note_text[:head_characters]
            + "\n\n[...NOTE TRUNCATED...]\n\n"
            + note_text[-tail_characters:]
        )

    return note_text[:max_characters]


def main():
    """Run extraction for every .txt file in data/input_notes."""
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"

    if not settings_path.exists():
        print(f"ERROR: Missing settings file: {settings_path}")
        return 1

    try:
        settings = load_yaml_file(settings_path)
    except yaml.YAMLError as error:
        print(f"ERROR: settings.yaml is not valid YAML: {error}")
        return 1

    logging_settings = settings.get("logging", {})
    setup_logging(
        project_path(
            logging_settings.get("error_file", "logs/pipeline_errors.log")
        )
    )

    files_settings = settings.get("files", {})
    input_settings = settings.get("input", {})
    output_settings = settings.get("output", {})
    batch_settings = settings.get("batch", {})
    failure_settings = settings.get("failure_handling", {})

    try:
        prompt_template = load_text_file(
            project_path(files_settings.get("prompt_file", "config/prompt.txt"))
        )
        vocabulary = load_yaml_file(
            project_path(files_settings.get("vocabulary_file", "config/vocabulary.yaml"))
        )
        csv_columns = load_json_file(
            project_path(
                files_settings.get("csv_columns_file", "config/csv_columns.json")
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as error:
        logging.error("Could not load configuration files: %s", error)
        print(f"ERROR: Could not load configuration files: {error}")
        return 1

    if "note_id" not in csv_columns:
        print("ERROR: csv_columns.json must contain a note_id field.")
        return 1

    input_folder = project_path(
        input_settings.get("folder", "data/input_notes")
    )
    output_folder = project_path(
        output_settings.get("folder", "data/output_csv")
    )
    output_csv = output_folder / output_settings.get(
        "filename", "extractions.csv"
    )

    input_folder.mkdir(parents=True, exist_ok=True)
    file_extension = input_settings.get("file_extension", ".txt").lower()
    note_files = sorted(
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() == file_extension
    )

    if not note_files:
        print(f"No {file_extension} files found in {input_folder}")
        return 0

    if output_settings.get("overwrite_existing_csv", False) and output_csv.exists():
        output_csv.unlink()

    if batch_settings.get("skip_existing_note_ids", True):
        existing_note_ids = get_existing_note_ids(output_csv)
    else:
        existing_note_ids = set()

    failed_log_path = project_path(
        logging_settings.get("failed_notes_file", "logs/failed_notes.log")
    )
    fieldnames = list(csv_columns.keys())
    input_encoding = input_settings.get("encoding", "utf-8")

    processed = 0
    skipped = 0
    failed = 0

    print(f"Found {len(note_files)} note(s).")
    print(f"Output CSV: {output_csv}")

    for note_path in note_files:
        note_id = note_path.stem

        if note_id in existing_note_ids:
            skipped += 1
            print(f"Skipped: {note_id}")
            continue

        try:
            note_text = load_text_file(note_path, input_encoding)

            if not note_text:
                raise ValueError("Note is empty.")

            note_text = truncate_note(note_text, settings)

            extraction = extract_note(
                note_text=note_text,
                note_id=note_id,
                settings=settings,
                prompt_template=prompt_template,
                vocabulary=vocabulary,
                csv_columns=csv_columns,
            )

            validated_extraction = validate_extraction(
                extraction=extraction,
                csv_columns=csv_columns,
                settings=settings,
                vocabulary=vocabulary,
                note_id=note_id,
                note_text=note_text,
            )

            row = make_csv_row(
                extraction=validated_extraction,
                csv_columns=csv_columns,
                settings=settings,
                note_id=note_id,
            )

            append_csv_row(
                csv_path=output_csv,
                row=row,
                fieldnames=fieldnames,
                encoding=output_settings.get("encoding", "utf-8"),
            )

            existing_note_ids.add(note_id)
            processed += 1
            print(f"Processed: {note_id}")

        except Exception as error:
            failed += 1
            logging.exception("Failed note_id=%s", note_id)
            save_failed_note(failed_log_path, note_id, str(error))
            print(f"Failed: {note_id} | {error}")

            if failure_settings.get("write_default_row_on_failure", True):
                fallback_row = make_csv_row(
                    extraction=csv_columns,
                    csv_columns=csv_columns,
                    settings=settings,
                    note_id=note_id,
                )
                append_csv_row(
                    csv_path=output_csv,
                    row=fallback_row,
                    fieldnames=fieldnames,
                    encoding=output_settings.get("encoding", "utf-8"),
                )
                existing_note_ids.add(note_id)

            if not failure_settings.get("continue_on_invalid_json", True):
                break

    print("\nPipeline complete.")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

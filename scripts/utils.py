from pathlib import Path
import json
import yaml


# Project root is the folder above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(relative_path):
    """Convert a project-relative path into an absolute path."""
    return PROJECT_ROOT / relative_path


def ensure_folder(folder_path):
    """Create a folder and any missing parent folders."""
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_text_file(file_path, encoding="utf-8"):
    """Read a text file and remove leading and trailing whitespace."""
    path = Path(file_path)

    with path.open("r", encoding=encoding) as file:
        return file.read().strip()


def load_json_file(file_path):
    """Read and return a JSON file."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml_file(file_path):
    """Read and return a YAML file."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_text_file(file_path, content, encoding="utf-8"):
    """Write text content to a file, creating parent folders if needed."""
    path = Path(file_path)
    ensure_folder(path.parent)

    with path.open("w", encoding=encoding) as file:
        file.write(str(content))


def save_json_file(file_path, data):
    """Write a dictionary or list to a formatted JSON file."""
    path = Path(file_path)
    ensure_folder(path.parent)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def get_note_id(note_path):
    """Use a note filename without its extension as the note ID."""
    return Path(note_path).stem


def get_note_files(input_folder, file_extension=".txt"):
    """Return sorted note files with the configured file extension."""
    folder = ensure_folder(input_folder)

    return sorted(
        file_path
        for file_path in folder.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() == file_extension.lower()
    )


def clean_text(value):
    """Convert a value to clean one-line text."""
    if value is None:
        return ""

    return str(value).strip().replace("\\n", " ")


def flatten_csv_value(value, separator=", "):
    """
    Convert a value into text suitable for one CSV cell.

    Lists are joined with the specified separator.
    None becomes an empty string.
    """
    if value is None:
        return ""

    if isinstance(value, list):
        cleaned_values = [
            clean_text(item)
            for item in value
            if item is not None and clean_text(item)
        ]
        return separator.join(cleaned_values)

    return clean_text(value)


def truncate_text(text, max_characters):
    """Trim text to a maximum number of characters."""
    if text is None:
        return ""

    text = str(text)

    if len(text) <= max_characters:
        return text

    return text[:max_characters]


def sanitize_filename(value):
    """Create a safe filename from a note ID or arbitrary text."""
    allowed_characters = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._-"
    )

    cleaned_value = "".join(
        character if character in allowed_characters else "_"
        for character in str(value)
    )

    return cleaned_value.strip("_") or "unnamed"

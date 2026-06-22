from datetime import datetime, timezone
from pathlib import Path
from typing import Literal



def parse_tes_indices(tes_indices: int | list[int] | Literal["all"], n_tes: int) -> list[int]:
    """
    Parse tes_indices and return the TES indices to process.

    If tes_indices is "all", return all TES indices from 0 to n_tes - 1.
    If tes_indices is an integer, return only that TES index.
    If tes_indices is a list of integers, return those TES indices.
    """

    if tes_indices == "all":
        tes_indices = list(range(n_tes))

    elif isinstance(tes_indices, list):
        tes_indices = [int(tes) for tes in tes_indices]

    else:
        tes_indices = [int(tes_indices)]

    invalid_tes_indices = [tes for tes in tes_indices
                           if tes < 0 or tes >= n_tes]

    if invalid_tes_indices:
        raise IndexError(f"Invalid TES indices {invalid_tes_indices}; dataset has {n_tes} TES "
                         f"with valid indices from 0 to {n_tes - 1}.")

    return tes_indices

def _resolve_path(path: Path,
                  root: Path) -> Path:
    """
    Resolve a path with respect to root if it is relative.
    """

    path = path.expanduser()
    root = root.expanduser().resolve()

    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()

def _to_value(value, unit):
    """
    Convert an astropy Quantity to a plain float in the requested unit.
    If value is already a plain number, return it unchanged.
    """

    if hasattr(value, "to_value"):
        return value.to_value(unit)

    return value

def parse_utc_datetime(value: str | datetime) -> datetime:
    """
    It takes the dataset name in string or datetime and parses it in
    datetime timezone-aware in UTC
    """

    if isinstance(value, datetime):
        dt = value
    else:
        # it transforms the Z suffix in an explicit offset ISO
        # EX: "2026-03-11T16:48:15Z" -> "2026-03-11T16:48:15+00:00"
        # Z means UTC
        text = str(value).strip().replace("Z", "+00:00")

        # it removes the dataset name which usually is encoded after the double underscores
        # EX: "2026-03-11_16.48.15__SkyDip"
        # and keeps just "2026-03-11_16.48.15"
        if "__" in text:
            text = text.split("__", 1)[0]

        # it converts the dataset name format in ISO format: YYYY-MM-DDTHH:MM:SS
        if "_" in text and "T" not in text:
            date_part, time_part = text.split("_", 1)
            text = f"{date_part}T{time_part.replace('.', ':')}"

        # it converts and ISO sting in a datetime object
        dt = datetime.fromisoformat(text)

    # it transforms the final result in UTC
    # EX: if "2026-03-11T16:48:15" becomes datetime(2026, 3, 11, 16, 48, 15)
    # dt.replace(tzinfo=timezone.utc) gives datetime(2026, 3, 11, 16, 48, 15, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    # if the timezone is present but the timezone is in UTC+1, it becomes UTC
    # EX: "2026-03-11T17:48:15+01:00" -> "2026-03-11T16:48:15+00:00"
    return dt.astimezone(timezone.utc)


def parse_ground_temperature_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        try:
            dt = datetime.strptime(text, "%H:%M:%S %d-%m-%Y")
        except ValueError:
            dt = parse_utc_datetime(text)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def date_string_from_datetime(value: str | datetime) -> str:
    return parse_utc_datetime(value).strftime("%Y-%m-%d")


if __name__ == "__main__":

    test_values = [
        "2026-03-11T16:48:15",
        "2026-03-11T16:48:15Z",
        "2026-03-11_16.48.15",
        "2026-03-11_16.48.15__SkyDip",
    ]

    print("UTC datetime parsing")
    for value in test_values:
        parsed = parse_utc_datetime(value)
        print(f"  {value!r} -> {parsed.isoformat()}")

    ground_test_values = [
        "16:46:07 11-03-2026",
        "2026-03-11T16:46:07",
        "2026-03-11_16.46.07",
    ]

    print("\nGround-temperature datetime parsing")
    for value in ground_test_values:
        parsed = parse_ground_temperature_datetime(value)
        print(f"  {value!r} -> {parsed.isoformat()}")

    dataset_name = "2026-03-11_16.48.15__SkyDip"
    print("\nDate string")
    print(f"  {dataset_name!r} -> {date_string_from_datetime(dataset_name)}")
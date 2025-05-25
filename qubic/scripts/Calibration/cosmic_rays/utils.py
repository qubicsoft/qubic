import csv
import json


def get_tes_asic_from_index(tes: int) -> tuple[int, int]:
    """
    Returns the ASIC and the local TES index within that ASIC.

    Parameters
    ----------
    tes: int
        Global TES index in the range [0, 255].


    Returns
    ------
    tuple[int, int]
        A tuple containing the local TES index (in the range [1, 128]) and its corresponding ASIC number.
    """

    # If TES index (tes+1) is less than 129, then the TES belongs to ASIC 1,
    # otherwise, adjust the index for ASIC 2.
    return (tes + 1, 1) if tes + 1 < 129 else (tes - 128 + 1, 2)


def write_results(fname: str,
                  data: dict,
                  indent: str | int | None = 2,
                  fmt: str = "json",
                  header: list[str] = None):
    """
    Writes a dictionary to a .json or .csv file.

    Parameters
    ---------
    fname: str
        The file name to which the dictionary should be saved.
    data: dict
        The dictionary containing data to be saved.
    indent: str | int | None
        The indentation level for the output file (used for JSON formatting).
    fmt: str
        The file format to use for saving ('json' or 'csv').
    header: list[str]
        The header row for the CSV file (used if fmt is 'csv').
    """

    with open(fname, "w", encoding='utf8') as fout:

        if fmt == "json":
            json.dump(data, fout, indent=indent)

        elif fmt == "csv":
            # Create a dictionary where keys are (ASIC, TES) tuples and values are
            # the averaged value of the time constant along with its sigma.
            tes_asic_data = {get_tes_asic_from_index(key)[::-1]: value for key, value in data.items()}

            # Sort the dictionary items by key (ASIC, TES) for consistent CSV ordering.
            sorted_data = sorted(tes_asic_data.items())

            # initialize writer object
            writer = csv.writer(fout)
            writer.writerow(header)

            # writes on the csv file
            for key, value in sorted_data:
                writer.writerow([*key, *map(lambda x: round(x, 4), value)])

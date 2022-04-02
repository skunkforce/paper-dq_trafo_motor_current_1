import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parents[1]
MEASUREMENTZIP = ROOT_DIR.joinpath("Messdaten.zip")
DATAFOLDER = ROOT_DIR.joinpath("tmp").joinpath("Asynchron Messungen PROLAB")
print(DATAFOLDER.exists())


def unpack_zipfile(zip_file=MEASUREMENTZIP, extract_dir: str = "./tmp"):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def unpack_measurement_data_if_needed():
    if not DATAFOLDER.exists():
        unpack_zipfile()


def extract_pico_csv_to_pd(csv_file) -> pd.DataFrame:
    result = pd.read_csv(csv_file,
                         sep=";",
                         decimal=",",
                         skiprows=[0, 2],
                         index_col=0,
                         )
    result = result.replace(["∞", "-∞"], [1.0, -1.0])
    result = result.applymap(lambda x: x if isinstance(x, (int, float)) else float(x.replace(",", ".")))
    result.columns = ["A", "B", "C", "D", "E", "F", "G"]
    result.index.names = ["Zeit"]
    result = result.astype(float)
    result.name = Path(csv_file).name
    return result


def extract_all_measurements() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (extract_pico_csv_to_pd(DATAFOLDER.joinpath("Nullmessung.csv")),
            extract_pico_csv_to_pd(DATAFOLDER.joinpath("unwucht1.csv")),
            extract_pico_csv_to_pd(DATAFOLDER.joinpath("unwucht2.csv")),
            extract_pico_csv_to_pd(DATAFOLDER.joinpath("unwucht3.csv")))

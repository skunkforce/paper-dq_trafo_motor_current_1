"""Module that contains the Data class and methods that is
used to read data from disk and store them as Pandas.DataFrame.
https://raw.githubusercontent.com/golgor/ps-signal/github_actions/ps_signal/signals/data.py
"""
import pandas as pd
import sys
import xlrd
import copy


def picoscope_data_loader(filename: str) -> pd.DataFrame:
    """Custom file loader for loading a file from disk. This file
    loader is made for loading files exported from picoscope as
    a .csv.

    Args:
        filename (str): The path to the file that should be imported.

    Returns:
        pd.DataFrame: A pandas.DataFrame containing all the data.

    Important:
        The supported file format for this loader is according to below.
        This is the standard when exporting a .csv from PicoScope Software.
        Note the ' ; ' as delimiter and ' , ' as decimal point.

        .. code-block:: python

            Tid;Kanal A
            (ms);(mV)

            -200,00016156;0,20752580
            -200,00004956;0,52491830
            ...
    """
    # Formatting of file is separated by ";" and decimals using ","
    # First two rows are headers.
    try:
        data = pd.read_csv(filename, sep=";", decimal=",", skiprows=[0, 2])
    except (FileNotFoundError, xlrd.biffh.XLRDError, Exception) as error:
        sys.exit(error)
    else:
        data.columns = ["time", "acc"]
        return data


class Data:
    """Class used for storing the data and important parameters.

    Args:
        loader (function): A function to use as a file importer.
            Defaults to picoscope_data_loader.
    """
    def __init__(self, loader=picoscope_data_loader) -> None:
        self._loader = loader
        self._data = None
        self._trigger_offset = None

    def load(self, data_path: str, remove_offset: bool = True) -> None:
        """Method used to load the actual file from disk into memory.
        Also calculates important parameters such as sampling frequency,
        sampling period and memory usage.

        Args:
            data_path (str): Path to or name of the file to be loaded.
            remove_offset (bool, optional): Set this to remove offset on
                the data. If a trigger offset was used during data collection,
                the first sample can have the time stamp -200ms instead of 0ms.
                Defaults to True.
        """
        try:
            self._data = self._loader(data_path)
        except Exception as error:
            print(error)

        self._size = len(self._data)
        self._frequency_hz = _calculate_sampling_frequency(self._data)
        self._period = 1 / self._frequency_hz
        self._memory_usage = self._data.memory_usage(index=True, deep=True)

        # With for example pre-trigger, the data starts from for example
        # -200ms. By substracting with the first value, the offset is removed.
        if remove_offset:
            self._trigger_offset = self._data.time.iloc[0]
            self._data.time -= self._data.time.iloc[0]

    @property
    def data(self) -> pd.DataFrame:
        """The imported data stored as a pd.DataFrame."""
        return self._data

    @property
    def size(self) -> int:
        """The row count of the imported data."""
        return self._size

    @property
    def frequency_hz(self):
        """The calculated sampling frequency."""
        return self._frequency_hz

    @property
    def period(self):
        """The calculated period, i.e. the inverse
        of the sampling frequency."""
        return self._period

    @property
    def memory_usage(self) -> pd.Series:
        """The memory used by the imported data.
        Stored as a pd.Series with one entry per column."""
        return self._memory_usage

    @property
    def memory_usage_mb(self) -> int:
        """The memory used by the imported data.
        Calculated to show total size in megabytes."""
        memory_mb = round(sum(self._memory_usage / 1000 ** 2), 3)
        return int(memory_mb)

    @property
    def memory_usage_kb(self) -> int:
        """The memory used by the imported data.
        Calculated to show total size in kilobytes."""
        memory_kb = round(sum(self._memory_usage / 1000), 3)
        return int(memory_kb)


def slice_data(data: Data, start_ms: int = None, end_ms: int = None) -> Data:
    """Function that takes a Data object and slice the data into a subset.
    Can be used if there is an interest only for a small part of the data.

    Args:
        data (Data): A Data object to be sliced.
        start_ms (int, optional): Where to start the slicing, given in ms.
            if None, the slice will start from the first sample.
            Defaults to None.
        end_ms (int, optional): Where to stop the slice, given in ms.
            If None, the slice will continue until the end of the data.
            Defaults to None.

    Returns:
        Data: Returns a data object that is a subset of the input.
    """

    """
    :return: Returns a data object that is a subset of the input
    :rtype: Data
    """
    if start_ms is None:
        start_ms = 0

    if end_ms is None:
        end_ms = data.size

    start_sample_count = round((start_ms / 1000) * data.frequency_hz)
    end_sample_count = round((end_ms / 1000) * data.frequency_hz)

    # Creating a copy to make sure there is two separate data sets,
    # i.e. not two object with references to the same data.
    # Using iloc from Pandas DataFrame object to slice the data.
    new_copy = copy.deepcopy(data)
    new_copy._data = new_copy.data.iloc[start_sample_count: end_sample_count]

    return new_copy


def _calculate_sampling_frequency(data: Data) -> int:
    """Function used to calculate the sampling frequency and making
    sure that the sampling frequency is constant. Without a constant
    frequency it will not be able to run FFT analysis on the signal.

    Args:
        data (Data): Input is a object of the Data class.

    Returns:
        int: Returns the sampling frequency as calculated from
        the time difference between the samples.
    """
    # Calculate a pandas series with the difference between all elements.
    diff = data.time.diff()[1:]

    # If the standard deviation is "high", the sampling rate is not consistent.
    # Without a consistent sampling frequency, a FFT will not be accurate.
    # Maximum std is for now an arbitrary number i.e. estimated based
    # on current available data.
    if not diff.std() < 1e-6:
        print("\nInconsistent sampling frequency found. \
              FFT will not be accurate!\n")

    # Mean value of the difference is the sampling frequency.
    # Division by 1000 due to time stored in ms and not seconds.
    mean_diff = round(sum(diff) / len(diff), 9) / 1000
    return round(1 / mean_diff)
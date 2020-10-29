import os
import sys
import csv
import json
import numpy as np
from ml.common.paths import DATA_DIR


_TRAINING_DATA_PATH = os.path.join(DATA_DIR, "train")
_TEST_DATA_PATH = os.path.join(DATA_DIR, "test")
_HTC_PATH = os.path.join(_TRAINING_DATA_PATH, "htc.csv")


def make_X_and_y(
    test: bool = False,
    timestep: int = 30,
    days: int = 28,
    repeats: int = 4,
    seed: int = 0,
    building_types: list = None,
    htc_upper_bound: float = None,
    htc_lower_bound: float = None,
    scale_inputs: bool = False,
    scale_targets: bool = False
):
    """
    Make an array of input features and an array of target features.

    X (input features) will look something like:
    [
        [
            [T_indoor_1, T_outdoor_1, gas_1, elec_1]
            [T_indoor_2, T_outdoor_2, gas_2, elec_2]
            ...
            [T_indoor_n, T_outdoor_n, gas_n, elec_n]
        ],
        .
        .
        .
    ]

    y (target features) will just be an array of HTCs.

    Naturally, X and y must have the same outer dimension and the dwellings must appear in the same
    order in both.

    We will round the the values in each series in X to the number of decimal places which we are
    most likely to get the real data to.
    - Indoor temp: 1dp
    - Outdoor temp: 1dp
    - Gas kwh: 2dp
    - Elec kwh: 3dp

    Args:
        test (bool):             Default = False. Whether to return test data or training data.
        timestep (int):          Default = 30. Data timestep in minutes.
        days (int):              Default = 28. Number of days of data to produce.
        repeats (int):           Default = 4. Number of samples to use for each house. These will
            be taken from different randomly chosen time windows.
        seed (int):              Default = 0. Seed for random number generator.
        building_types (list):   Default = None. List of buiding types to filter by.
        htc_upper_bound (float): Default = None. Only include houses with mean htc less than this.
        htc_lower_bound (float): Default = None. Only include houses with mean htc more than this.
        scale_inputs (bool):     Default = False. Whether or not to scale the inputs into [0,1].
        scale_targets (bool):    Default = False. Whether or not to scale the targets into [0,1].

    Returns:
        X (np.ndarray), y (np.ndarray): The input and target features
    """
    data_path = _TEST_DATA_PATH if test else _TRAINING_DATA_PATH
    indoor = np.load(os.path.join(data_path, f"indoor_temp.npy")).round(1)
    outdoor = np.load(os.path.join(data_path, f"outdoor_temp.npy")).round(1)
    gas = np.load(os.path.join(data_path, f"gas_kwh.npy")).round(2)
    elec = np.load(os.path.join(data_path, f"elec_kwh.npy")).round(3)

    # Get X in the right shape (channels last).
    # The transpose bit means it permutes the dimensions 0,1,2 -> 1,2,0
    # So the dimensions are (houses, timesteps, channels)
    X = np.array([indoor, outdoor, gas, elec]).transpose((1,2,0))

    # This is basically the contents of htc.csv, but only the `htc_<n>` columns
    htcs_array = []
    with open(_HTC_PATH, "r") as f:
        reader = csv.reader(f)
        row = next(reader)
        htcs_start_ind = row.index("htc_0")

        for row in reader:
            htcs_array.append([float(h) for h in row[htcs_start_ind:]])

    htcs_array = np.array(htcs_array)

    # This is a list of all the indices of the houses which meet our building type and htc conditions
    filtered_indices = _make_sample_filter_indices(X, test, building_types, htc_upper_bound, htc_lower_bound)
    # Filter X and htcs_array by it
    X = X[filtered_indices]
    htcs_array = [htcs_array[i] for i in filtered_indices]

    # Slice X to the correct number of days, pick the HTC value corresponding time period
    # and make repeats if specified.
    rng = np.random.default_rng(seed=seed)
    newX = []
    newy = []
    no_timesteps = days * 1440 // timestep # 1440 = number of minutes in a day
    total_timesteps = X.shape[1]

    # Can't start less than 28 days from the end of the data
    latest_start_point = 28 * 1440 // timestep

    # Make a X and y for the specified number of days and repeats
    for i, house in enumerate(X):
        for repeat in range(repeats):
            start = rng.integers(0, total_timesteps - latest_start_point)
            end = start + no_timesteps
            newX.append(house[start:end,:])

            # Add the htc for the corresponding time period to newy
            y_start = int(start // (1440 / timestep))
            newy.append(htcs_array[i][y_start])

    # Scale inputs and targets if necessary
    if scale_inputs:
        # Scale the features into [0,1] before returning
        scale_factors = get_feature_scale_factors()
        newX = np.array(newX)
        X = np.array([
            newX[:,:,0] / scale_factors["indoor_temp"],
            newX[:,:,1] / scale_factors["outdoor_temp"],
            newX[:,:,2] / scale_factors["gas_kwh"],
            newX[:,:,3] / scale_factors["elec_kwh"],
        ]).transpose((1,2,0))
    else:
        X = np.array(newX)

    if scale_targets:
        # Scale the features into [0,1] before returning
        scale_factors = get_feature_scale_factors()
        y = np.array(newy) / scale_factors["htc"]
    else:
        y = np.array(newy)

    return X, y


def make_sqm_X(test: bool = False, repeats: int = 4, scale: bool = False):
    """Make array of sq. m values for the houses, optionally repeating them as required."""
    sqm_list = _sqm_from_csv(test)
    sqm_X = np.array(sqm_list)

    if repeats > 1:
        sqm_X = np.repeat(sqm_X, repeats)

    if scale:
        # Scale the floor_areas into [0,1] before returning
        scale = get_feature_scale_factors()["floor_area"]
        sqm_X = sqm_X / scale

    # Reshape so that each element is a singleton array
    return sqm_X.reshape((-1, 1))


def make_htc_proxy_X(X: np.ndarray):
    """
    Makes HTC proxy values from data series.
    The value of the HTC proxy is sum(gas) / mean(in_temp - out_temp).
    """
    return np.array([[np.sum(x[:,2]) / np.sum(x[:,0] - x[:,1])] for x in X])


def make_feature_scale_factors():
    """Saves a dictionary of features->scale_factors such that each feature array can be mapped into [0,1]."""
    X, y = make_X_and_y()
    sqm = make_sqm_X()
    scale_factors = {
        "indoor_temp": np.max(X[:,:,0]),
        "outdoor_temp": np.max(X[:,:,1]),
        "gas_kwh": np.max(X[:,:,2]),
        "elec_kwh": np.max(X[:,:,3]),
        "floor_area": np.max(sqm),
        "htc": np.max(y),
    }

    with open(os.path.join(_TRAINING_DATA_PATH, "scalefactors.json"), "w+") as f:
        json.dump(scale_factors, f)


def get_feature_scale_factors():
    """
    Returns a dictionary mapping feature name to scale factor. Feature names are:
        indoor_temp
        outdoor_temp
        gas_kwh
        elec_kwh
        floor_area
        htc
    """
    with open(os.path.join(_TRAINING_DATA_PATH, "scalefactors.json"), "r") as f:
        factors = json.load(f)
        return factors


def unscale_outputs(outputs: np.ndarray):
    """Rescale HTC predictions so that they are of the right order of magnitude."""
    scale_factors = get_feature_scale_factors()
    return outputs * scale_factors["htc"]


def _make_sample_filter_indices(
    X: np.ndarray,
    test: bool = False,
    building_types: list = None,
    htc_upper_bound: float = None,
    htc_lower_bound: float = None
):
    """
    All of the files containing input features or HTCs should have the houses in the same order.
    If we want to generate an X and y with just a subset of the training data, it is useful
    to have a list of the indices of all the houses which make it through our filter.
    """
    all_indices = list(range(X.shape[0]))

    print(">>>", building_types)

    if building_types and len(building_types) > 0:
        building_types_list = _building_types_from_csv(test)
        all_indices += [i for i, bt in enumerate(building_types_list) if bt in building_types]

    if htc_lower_bound:
        mean_htcs = _mean_htc_from_csv(test)
        htc_lower_filtered_indices = np.where(np.array(mean_htcs) >= htc_lower_bound)[0].tolist()

        if len(all_indices) > 0:
            all_indices = list(set(all_indices) & set(htc_lower_filtered_indices))
        else:
            all_indices = htc_lower_filtered_indices

    if htc_upper_bound:
        mean_htcs = _mean_htc_from_csv(test)
        htc_upper_filtered_indices = np.where(np.array(mean_htcs) <= htc_upper_bound)[0].tolist()

        if len(all_indices) > 0:
            all_indices = list(set(all_indices) & set(htc_upper_filtered_indices))
        else:
            all_indices = htc_upper_filtered_indices

    return all_indices


def _building_types_from_csv(test: bool = False):
    """Read all the building types from data/train/building_type.csv into a list."""
    data_path = _TEST_DATA_PATH if test else _TRAINING_DATA_PATH
    with open(os.path.join(data_path, "building_type.csv")) as f:
        return [row["building_type"] for row in csv.DictReader(f)]


def _sqm_from_csv(test: bool = False):
    """Read all the sq. m values from data/train/sqm.csv into a list."""
    data_path = _TEST_DATA_PATH if test else _TRAINING_DATA_PATH
    with open(os.path.join(data_path, "sqm.csv")) as f:
        return [float(row["sqm"]) for row in csv.DictReader(f)]


def _mean_htc_from_csv(test: bool = False):
    """Read all the mean HTC values from data/train/htc.csv into a list."""
    data_path = _TEST_DATA_PATH if test else _TRAINING_DATA_PATH
    with open(os.path.join(data_path, "htc.csv")) as f:
        return [float(row["mean_htc"]) for row in csv.DictReader(f)]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Expected at least one argument.")
    else:
        if sys.argv[1] == "MAKE_FEATURE_SCALE_FACTORS":
            make_feature_scale_factors()

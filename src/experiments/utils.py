# -*- coding: utf-8 -*-
"""
@author: jonas

Utility functions for hand gesture data preprocessing, quaternion-based
transformations, data analysis, and model inference.
"""

import os
import pandas as pd
import numpy as np
from itertools import chain, combinations
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import models
from scipy.signal import resample
import models_jit
# -------------------------------------------------------------------------
# Global constants and metadata
# -------------------------------------------------------------------------

num_sensors = 8
object_list = ['stick', 'ball', 'screw', 'box', 'knife', 'glass', 'assembly', 'disk', 'rest']

# Data column templates for board and glove data
board_data_columns = [f'board_data_{i}' for i in range(8)]
glove_data_columns = [f'SensorID_{i}_{axis}' for i in range(25) for axis in ['x', 'y', 'z']]

# All expected columns for the main dataset
columns = (
    glove_data_columns +
    ['glove_app_time', 'glove_timestamps'] +
    [f'board_data_{i}' for i in range(8)] +
    ['board_timestamps', 'palm_data', 'palm_timestamps', 'hand_active',
     'target_object', 'trial_success', 'block', 'participant',
     'trial_number', 'rep_number', 'attempt_number']
)

colors = {
    0: '#2bad70',           # #0072B2 - bold and deep
    1: '#416897',         # #E69F00 - warm and vibrant
    3: '#84cfed',          # #009E73 - calm and nature-inspired
    2: '#afc410',           # #999999 - neutral and balanced
    4: '#EDA985',            # #D55E00 - strong and attention-grabbing
    5: '#96368b',         # #CC79A7 - soft and distinctive
    6: '#d9328a',           # #56B4E9 - bright and clear
    7: 'black',         # #F0E442 - bright and energetic
    8: 'teal',           # #1B9E77 - modern and refreshing
    9: 'pink'            # #E78AC3 - playful and elegant
}

# Sensor permutations for various alignment or symmetry operations
perm0 = [0, 1, 2, 3, 4, 5, 6, 7]
perm1 = [0, 1, 2, 3, 4, 5, 6, 7]
perm2 = perm1[::-1]
perm3 = [3, 2, 1, 0, 7, 6, 5, 4]
perm4 = perm3[::-1]
perms = [perm0, perm1, perm2, perm3, perm4]


def generate_object_subsets():
    nums = list(range(8))  # numbers 0–7
    subsets = []
    n = len(nums)
    
    # There are 2^8 possible subsets
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        subset.append(8)
        if len(subset) >= 2:
            subsets.append(subset)
    
    return subsets


def extract_classes(X, y, class_list):
    """
    Return only samples whose labels are in class_list.

    Parameters:
        X : numpy array of shape (n_samples, n_features)
        y : numpy array of shape (n_samples,)
        class_list : list or set of labels to retain

    Returns:
        X_filtered, y_filtered : filtered arrays
    """
    class_list = set(class_list)  # faster membership check
    mask = np.isin(y, list(class_list))
    return X[mask], y[mask]

# -------------------------------------------------------------------------
# Basic signal processing and indexing utilities
# -------------------------------------------------------------------------
def compute_thumb_index_distance(data: np.ndarray) -> np.ndarray:
    """
    Compute the area of the triangle defined by 3 points in 3D space for each row of the array.
    
    Parameters:
        data (np.ndarray): shape (n, 9), where each row has 3 points as [x1,y1,z1,x2,y2,z2,x3,y3,z3]
    
    Returns:
        np.ndarray: shape (n,), areas for each triangle
    """
    if data.shape[1] != 6:
        raise ValueError("Each row must contain 9 elements (3 points in 3D)")

    # Extract points A, B, C
    A = data[:, 0:3]
    B = data[:, 3:6]

    # Compute vectors AB and AC
    AB = B - A

    area = 0.5 * np.linalg.norm(AB, axis=1)

    return area


def first_crossing_index(series: np.ndarray, threshold: float, steps: int) -> int:
    """
    Find the first index where the series crosses a threshold for a consecutive number of steps.

    Parameters
    ----------
    series : np.ndarray
        Input 1D signal array.
    threshold : float
        Threshold value to detect crossing.
    steps : int
        Number of consecutive samples above threshold to confirm crossing.

    Returns
    -------
    int
        Index of the first threshold crossing. Returns -1 if not found.
    """
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    above = series > threshold
    conv = np.convolve(above, np.ones(steps, dtype=int), mode='valid')
    idx = np.where(conv == steps)[0]
    return idx[0] if len(idx) > 0 else -1


def replace_with_highest(arr, target):
    """
    Replace all occurrences of `target` in array with the highest non-target value.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    target : scalar
        Value to be replaced.

    Returns
    -------
    np.ndarray
        Array with target values replaced.
    """
    if arr is None or arr.size == 0:
        return arr

    mask = arr != target
    if not np.any(mask):
        return arr  # All elements are the target value

    max_non_target = np.max(arr[mask])
    result = np.where(arr == target, max_non_target, arr)
    return result


def find_first_of_final_sequence(arr, last):
    """
    Find the first index of the final sequence of identical elements.

    Parameters
    ----------
    arr : np.ndarray
        Input array (1D or 2D).
    last : scalar or array-like
        Final element value(s) to match.

    Returns
    -------
    int
        Index where the final repeating sequence begins.
    """
    n = len(arr)
    if len(arr.shape) == 1:
        for i in range(n - 2, -1, -1):
            if arr[i] != last:
                return i + 1
        return 0
    else:
        for i in range(n - 2, -1, -1):
            if last not in arr[i]:
                return i + 1
        return 0


def find_switch_indices(arr):
    """
    Identify indices where values in an array change.

    Supports both 1D and 2D arrays.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D or 2D array.

    Returns
    -------
    np.ndarray
        Indices where value changes occur.
    """
    if arr.ndim == 1:
        if arr.size == 0:
            return np.array([], dtype=int)
        switches = arr[1:] != arr[:-1]
    elif arr.ndim == 2:
        if arr.shape[0] < 2:
            return np.array([], dtype=int)
        sorted_rows = np.sort(arr, axis=1)
        switches = np.any(sorted_rows[1:] != sorted_rows[:-1], axis=1)
    else:
        raise ValueError("Only 1D or 2D arrays are supported.")
    return np.where(switches)[0] + 1


def compute_distance_travelled(ts, target):
    """
    Compute normalized cumulative distance traveled toward a target point.

    Parameters
    ----------
    ts : np.ndarray
        Sequence of 3D positions.
    target : np.ndarray
        Target 3D point.

    Returns
    -------
    np.ndarray
        Cumulative distance as percentage of total distance.
    """
    distance = np.array([np.linalg.norm(element - target) for element in ts])
    distance = abs(distance)
    return distance


def compute_distance_travelled_in_percent(ts, target):
    """
    Compute normalized cumulative distance traveled toward a target point.

    Parameters
    ----------
    ts : np.ndarray
        Sequence of 3D positions.
    target : np.ndarray
        Target 3D point.

    Returns
    -------
    np.ndarray
        Cumulative distance as percentage of total distance.
    """
    end = target
    distance = [np.linalg.norm(element - end) for element in ts]
    distance = abs(np.diff(distance, axis=0))
    dist_cumsum = np.cumsum(distance)
    return dist_cumsum / dist_cumsum[-1]


def analyze_probability_distribution(prob_matrix):
    """
    Analyze probability distributions row-wise, sorting and computing cumulative sums.

    Parameters
    ----------
    prob_matrix : np.ndarray
        2D array of probabilities.

    Returns
    -------
    cumulative_sums : np.ndarray
        Cumulative sum of sorted probabilities.
    sorted_indices : np.ndarray
        Indices of sorted probabilities.
    """
    sorted_indices = np.argsort(-prob_matrix, axis=1)
    sorted_probs = np.take_along_axis(prob_matrix, sorted_indices, axis=1)
    cumulative_sums = np.cumsum(sorted_probs, axis=1)
    return cumulative_sums, sorted_indices


def compute_k_uncertainty(probs):
    """
    Compute k-step uncertainty metric (custom entropy measure).

    Parameters
    ----------
    probs : np.ndarray
        Probability matrix.

    Returns
    -------
    np.ndarray
        Computed uncertainty values.
    """
    cumsums, sets = analyze_probability_distribution(probs)
    entropy = np.empty(cumsums.shape)
    for k in range(cumsums.shape[1]):
        for n in range(cumsums.shape[0]):
            summe = np.sum(probs[n, sets[n, :k + 1]])
            rest = probs[n, sets[n, k + 1:]]
            dummy = np.zeros(cumsums.shape[1] - k)
            dummy[0] = summe
            dummy[1:] = rest
            entropy[n, k] = 1 - np.sum(dummy * np.log2(dummy)) / -np.log2(cumsums.shape[1] - k)
    return entropy


def create_pandas_frame():
    """
    Create an empty pandas DataFrame with predefined column structure.

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with expected columns.
    """
    df = pd.DataFrame(columns=columns)
    return df


# -------------------------------------------------------------------------
# Quaternion and vector transformations
# -------------------------------------------------------------------------

def quaternion_multiply(q1, q2):
    """
    Perform quaternion multiplication q1 * q2.

    Parameters
    ----------
    q1 : np.ndarray
        First quaternion [w, x, y, z].
    q2 : np.ndarray
        Second quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray
        Resulting quaternion from multiplication.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])


def rotate_vector_by_quaternion(v, q):
    """
    Rotate a 3D vector by a quaternion.

    Parameters
    ----------
    v : array-like, shape (3,)
        Vector to rotate.
    q : array-like, shape (4,)
        Quaternion [w, x, y, z] defining rotation.

    Returns
    -------
    np.ndarray
        Rotated 3D vector.
    """
    v_quat = np.array([0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    temp = quaternion_multiply(q, v_quat)
    result = quaternion_multiply(temp, q_conj)
    return result[1:4]


def rotate_point_by_quaternion(points, quaternion):
    """
    Rotate one or more 3D points by a quaternion using scipy Rotation.

    Parameters
    ----------
    points : np.ndarray
        Array of 3D points, shape (N, 3) or (3,).
    quaternion : np.ndarray
        Quaternion [x, y, z, w] (scipy expects this order).

    Returns
    -------
    np.ndarray
        Rotated point(s).
    """
    r = R.from_quat(quaternion)
    return r.apply(points)


def quaternion_between_vectors(v1, v2):
    """
    Compute quaternion representing rotation from vector v1 to v2.

    Parameters
    ----------
    v1, v2 : array-like, shape (3,)
        Input vectors.

    Returns
    -------
    np.ndarray
        Quaternion [w, x, y, z].
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if dot < -0.999999:  # Vectors are opposite
        axis = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1, axis)
        axis /= np.linalg.norm(axis)
        return np.array([0, axis[0], axis[1], axis[2]])  # 180° rotation

    s = np.sqrt((1 + dot) * 2)
    inv_s = 1 / s
    return np.array([s * 0.5, cross[0] * inv_s, cross[1] * inv_s, cross[2] * inv_s])


def normalize_vector(v):
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Normalized vector.

    Raises
    ------
    ValueError
        If vector norm is zero.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


def find_quaternion_for_cross_product_y_alignment(v1, v2, target_direction):
    """
    Compute quaternion that aligns cross(v1, v2) with a target direction.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Input 3D vectors.
    target_direction : np.ndarray
        Target direction vector to align cross product with.

    Returns
    -------
    np.ndarray
        Quaternion performing the alignment.
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    current_cross = np.cross(v1, v2)
    current_cross_normalized = normalize_vector(current_cross)
    return quaternion_between_vectors(current_cross_normalized, target_direction)


def align_in_plane(hand_data, plane_vec_1, plane_vec_2, normal_vec):
    """
    Aligns 3D hand data so that cross(plane_vec_1, plane_vec_2) aligns with normal_vec.

    Parameters
    ----------
    hand_data : np.ndarray
        Array of hand joint positions.
    plane_vec_1, plane_vec_2 : list[int]
        Index triplets specifying plane-defining vectors.
    normal_vec : np.ndarray
        Target normal vector for alignment.

    Returns
    -------
    np.ndarray
        Rotated/Aligned hand data.
    """
    frame = np.empty_like(hand_data)
    for i in range(len(hand_data)):
        qdiff = find_quaternion_for_cross_product_y_alignment(
            v1=hand_data[i, plane_vec_1],
            v2=hand_data[i, plane_vec_2],
            target_direction=normal_vec
        )
        qdiff = ensure_positive_real_part(qdiff)
        points = hand_data[i].reshape(-1, 3)
        rotated_points = np.empty(points.shape)
        for j in range(len(points)):
            rotated_points[j] = rotate_vector_by_quaternion(points[j], qdiff)
        frame[i] = rotated_points.flatten()
    return frame


def ensure_positive_real_part(quaternion):
    """
    Ensure quaternion real (scalar) part is positive.

    Parameters
    ----------
    quaternion : np.ndarray
        Quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray
        Quaternion with positive scalar part.
    """
    q = np.array(quaternion, dtype=float)
    if q[0] < 0:
        q = -q
    return q


def find_large_time_gaps(timestamps, threshold_ms):
    """
    Identify indices where timestamp gaps exceed a threshold.

    Parameters
    ----------
    timestamps : array-like
        Sequence of timestamps.
    threshold_ms : float
        Threshold gap (milliseconds).

    Returns
    -------
    np.ndarray
        Indices where large gaps occur.
    """
    timestamps = np.asarray(timestamps)
    diffs = np.abs(np.diff(timestamps))
    indices = np.where(diffs > threshold_ms)[0]
    return indices


def get_differential_quaternion(point_a, point_b):
    """
    Compute quaternion rotating point_a onto point_b.

    Parameters
    ----------
    point_a, point_b : np.ndarray
        3D unit vectors.

    Returns
    -------
    np.ndarray
        Quaternion representing rotation.
    """
    point_a = point_a / np.linalg.norm(point_a)
    point_b = point_b / np.linalg.norm(point_b)
    dot_product = np.dot(point_a, point_b)

    if np.allclose(point_a, point_b):
        return np.array([0, 0, 0, 1])

    if np.allclose(point_a, -point_b):
        perp_axis = np.array([1, 0, 0]) if abs(point_a[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(point_a, perp_axis)
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_quat()

    axis = np.cross(point_a, point_b)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.array([0, 0, 0, 1])
    axis /= axis_norm
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    quat = R.from_rotvec(angle * axis).as_quat()
    return quat


def rotate_into_axis(hand_data, indices, axis):
    """
    Rotate all frames so that a reference vector aligns with a given axis.

    Parameters
    ----------
    hand_data : np.ndarray
        3D hand positions per frame.
    indices : list[int]
        Index triplet specifying reference vector.
    axis : np.ndarray
        Target axis to align to.

    Returns
    -------
    np.ndarray
        Rotated data frames.
    """
    frame = np.empty_like(hand_data)
    for i in range(len(hand_data)):
        qdiff = get_differential_quaternion(hand_data[i, indices], axis)
        qdiff = ensure_positive_real_part(qdiff)
        points = hand_data[i].reshape(-1, 3)
        rotated_points = rotate_point_by_quaternion(points, qdiff)
        frame[i] = rotated_points.flatten()
    return frame


def preprocess_data_rotation(glove_data_subset):
    """
    Apply standard rotation preprocessing to glove data.

    Parameters
    ----------
    glove_data_subset : np.ndarray
        Glove data.

    Returns
    -------
    np.ndarray
        Rotated glove data.
    """
    glove_data_subset = rotate_into_axis(
        glove_data_subset, indices=[18, 19, 20], axis=np.array([0, 0, 1])
    )
    glove_data_subset = align_in_plane(
        glove_data_subset, plane_vec_1=[18, 19, 20],
        plane_vec_2=[33, 34, 35], normal_vec=np.array([0, 1, 0])
    )
    return glove_data_subset


# -------------------------------------------------------------------------
# Data reduction and selection utilities
# -------------------------------------------------------------------------

def generate_finger_subsets():
    """
    Generate all subsets of five fingers.

    Returns
    -------
    list[list[int]]
        List of subsets (as lists of indices).
    list[str]
        Corresponding string identifiers.
    """
    elements = [0, 1, 2, 3, 4]
    all_subsets = list(chain.from_iterable(
        combinations(elements, r) for r in range(len(elements) + 1)
    ))
    list_of_lists = [list(subset) for subset in all_subsets]
    list_of_strings = [''.join(str(num) for num in subset) for subset in all_subsets]
    return list_of_lists[1:], list_of_strings[1:]  # exclude empty set


def preprocess_data_reduction(glove_data_subset, number_of_fingers):
    """
    Reduce glove data to selected number of fingers.

    Parameters
    ----------
    glove_data_subset : np.ndarray
        Input glove data.
    number_of_fingers : int
        How many fingers to keep (1–5).

    Returns
    -------
    np.ndarray
        Reduced feature frame.
    """
    if glove_data_subset.ndim == 1:
        glove_data_subset = glove_data_subset.reshape(1, -1)

    frame = np.empty((glove_data_subset.shape[0], int(number_of_fingers * 3)))

    # Finger indices hardcoded according to sensor layout
    index_map = {
        5: [12, 27, 42, 57, 72],
        4: [12, 27, 42, 57],
        3: [12, 27, 42],
        2: [12, 27],
        1: [12],
    }

    for c, i in enumerate(index_map[number_of_fingers]):
        frame[:, c * 3:(c + 1) * 3] = glove_data_subset[:, i:i + 3]
    return frame


def preprocess_data_finger_config(glove_data_subset, finger_config):
    """
    Select and reorder finger data according to configuration.

    Parameters
    ----------
    glove_data_subset : np.ndarray
        Input glove data.
    finger_config : list[int]
        List of finger indices to include.

    Returns
    -------
    np.ndarray
        Reduced and reordered glove data.
    """
    if glove_data_subset.ndim == 1:
        glove_data_subset = glove_data_subset.reshape(1, -1)

    glove_data_subset = preprocess_data_reduction(glove_data_subset, 5)
    frame = np.empty((glove_data_subset.shape[0], int(len(finger_config) * 3)))

    for c, i in enumerate(finger_config):
        frame[:, c * 3:(c + 1) * 3] = glove_data_subset[:, i * 3:i * 3 + 3]
    return frame


# -------------------------------------------------------------------------
# File I/O utilities
# -------------------------------------------------------------------------

def load_dataframe_from_csv(filepath, dtype=None):
    """
    Load a pandas DataFrame from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    dtype : dict or None, optional
        Optional mapping of column names to data types.

    Returns
    -------
    pandas.DataFrame or None
        Loaded DataFrame, or None if an error occurred.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    try:
        if dtype:
            df = pd.read_csv(filepath, dtype=dtype)
        else:
            df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading DataFrame from CSV: {e}")
        return None


def create_prediction_dataframe():
    """
    Create a structured DataFrame for model predictions.

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with predefined columns for predictions.
    """
    columns = ['block', 'trial_number', 'rep_number']
    columns += [f'mean_pred_{i}' for i in range(9)]
    columns += [f'var_pred_{i}' for i in range(9)]
    columns += [f'likelihood_{i}' for i in range(9)]
    columns += ['noise_KF', 'noise_obs', 'window_length', 'grav_const']
    columns += [f'nc_pred_{i}' for i in range(9)]
    return pd.DataFrame(columns=columns)


def add_row_to_prediction_dataframe(df, phase, trial, repetition,
                                    mean_pred, var_pred, likelihoods,
                                    noise_KF, noise_obs, window_length,
                                    grav_const, nc_pred):
    """
    Append a single result row to a prediction DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction DataFrame.
    phase, trial, repetition : int
        Experiment identifiers.
    mean_pred, var_pred, likelihoods, nc_pred : list or np.ndarray
        Model outputs (length 9 each).
    noise_KF, noise_obs : float
        Noise parameters.
    window_length : int
        Integration window length.
    grav_const : float
        Gravitational constant used in the model.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame.
    """
    if not isinstance(mean_pred, (list, np.ndarray)) or len(mean_pred) != 9:
        raise ValueError("mean_pred must be a list or numpy array of length 9")
    if not isinstance(var_pred, (list, np.ndarray)) or len(var_pred) != 9:
        raise ValueError("var_pred must be a list or numpy array of length 9")
    if not isinstance(likelihoods, (list, np.ndarray)) or len(likelihoods) != 9:
        raise ValueError("likelihoods must be a list or numpy array of length 9")
    if not isinstance(nc_pred, (list, np.ndarray)) or len(nc_pred) != 9:
        raise ValueError("nc_pred must be a list or numpy array of length 9")

    mean_pred = list(mean_pred)
    var_pred = list(var_pred)
    likelihoods = list(likelihoods)
    nc_pred = list(nc_pred)

    new_entry = (
        [phase, trial, repetition] +
        mean_pred + var_pred + likelihoods +
        [noise_KF, noise_obs, window_length, grav_const] +
        nc_pred
    )
    df.loc[len(df)] = new_entry
    return df

def add_rows_to_prediction_dataframe_batch(df, phase, trial, repetition,
                                           means, vars, likelihoods, nc_preds,
                                           noise_KF, noise_obs, window_length, grav_const):
    """
    Append multiple result rows to a prediction DataFrame efficiently.
    """
    n_rows = len(means)
    
    # Validate once
    for arr, name in [(means, 'means'), (vars, 'vars'), 
                      (likelihoods, 'likelihoods'), (nc_preds, 'nc_preds')]:
        if len(arr) != n_rows or any(len(x) != 9 for x in arr):
            raise ValueError(f"{name} must be array of length-9 arrays")
    
    # Build all rows at once
    new_rows = [
        [phase, trial, repetition] +
        list(mean) + list(var) + list(ll) +
        [noise_KF, noise_obs, window_length, grav_const] +
        list(nc_p)
        for mean, var, ll, nc_p in zip(means, vars, likelihoods, nc_preds)
    ]
    
    return pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], 
                     ignore_index=True)


def save_dataframe_to_csv(df, filepath, index=False):
    """
    Save a DataFrame to CSV, creating directories if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save.
    filepath : str
        Output CSV path.
    index : bool, optional
        Whether to include the index column.
    """
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(filepath, index=index)
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")


# -------------------------------------------------------------------------
# Miscellaneous data transformation utilities
# -------------------------------------------------------------------------

def first_change_index(time_series, threshold):
    """
    Return the first index where the time series exceeds a threshold.

    Parameters
    ----------
    time_series : np.ndarray
        Input data.
    threshold : float
        Threshold value.

    Returns
    -------
    int or None
        Index of first change or None if no change found.
    """
    changes = np.where(np.abs(time_series) > threshold)[0]
    return changes[0] + 1 if changes.size > 0 else None


def transform_data_to_speed_profile(data, ts):
    """
    Compute speed profile (magnitude of derivative) from positional data.

    Parameters
    ----------
    data : np.ndarray
        Position data (1D or 2D).
    ts : np.ndarray
        Time stamps for gradient computation.

    Returns
    -------
    np.ndarray
        Speed profile array.
    """
    grad = np.gradient(data, ts, axis=0)
    if data.ndim == 1:
        speed = np.abs(grad)
    else:
        speed = LA.norm(grad, axis=1)
    return speed


def indices_bigger_than_threshold(arr, threshold):
    """
    Return indices of array elements greater than a threshold.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    threshold : float
        Threshold value.

    Returns
    -------
    list[int]
        List of indices.
    """
    return [index for index, value in enumerate(arr) if value > threshold]


# -------------------------------------------------------------------------
# Inference and model integration
# -------------------------------------------------------------------------

def run_inference_trial_phase(df_filename, target_df_filename, beta,
                              noise_KF, noise_obs, X, y, dt,
                              integration_window_length,
                              preprocessing_function_for_data_reduction):
    """
    Run inference for all trial phases and save results to CSV.

    Parameters
    ----------
    df_filename : str
        Source dataset file path.
    target_df_filename : str
        Output CSV file for predictions.
    beta : float
        Gravity parameter used in model.
    noise_KF, noise_obs : float
        Noise parameters for filters.
    X : np.ndarray
        Training data (features).
    y : np.ndarray
        Training labels.
    dt : float
        Time step.
    integration_window_length : int
        Integration window size.
    preprocessing_function_for_data_reduction : callable
        Function applied to reduce glove data dimensionality.

    Notes
    -----
    - Relies on `models.NearestCentroidClassifier` and `models.GravFromData`.
    - Writes predictions to a structured CSV.
    """
    nc = models.NearestCentroidClassifier()
    nc.integration_window = integration_window_length
    nc.fit(X, y)

    if os.path.exists(target_df_filename):
        print(f"The file '{target_df_filename}' exists.")
    else:
        df = load_dataframe_from_csv(df_filename)
        pred_df = create_prediction_dataframe()

        for phase in range(df['block'].max() + 1):
            for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
                for rep_counter in range(df[(df['block'] == phase) &
                                            (df['trial_number'] == trial_number)]['rep_number'].max() + 1):
                    for noise_KF in [noise_KF]:
                        for noise_obs in [noise_obs]:
                            for beta in [beta]:
                                df_att = df[(df['block'] == phase) &
                                            (df['trial_number'] == trial_number) &
                                            (df['rep_number'] == rep_counter)]

                                glove_data = df_att[glove_data_columns].to_numpy()
                                glove_time = df_att['glove_timestamps'].to_numpy()

                                print('Running inference on:', target_df_filename)
                                print('Phase', phase, 'Trial', trial_number, 'Rep', rep_counter)

                                x0 = preprocessing_function_for_data_reduction(glove_data[0])
                                grav_model = models.GravFromData(
                                    observed_gestures=X,
                                    labels=y,
                                    noise_KF=noise_KF,
                                    noise_obs=noise_obs,
                                    x0=x0[0],
                                    dt=dt,
                                    integration_window_length=integration_window_length,
                                    beta=beta
                                )

                                obs = preprocessing_function_for_data_reduction(glove_data)
                                nc_pred = nc.run_sequence(obs)
                                means, var, ll = grav_model.run_sequence(
                                    observations=obs, t_obs=glove_time
                                )

                                for mean, variance, logl, nc_p in zip(means, var, ll, nc_pred):
                                    add_row_to_prediction_dataframe(
                                        df=pred_df,
                                        phase=phase,
                                        trial=trial_number,
                                        repetition=rep_counter,
                                        mean_pred=mean,
                                        var_pred=variance,
                                        noise_KF=noise_KF,
                                        noise_obs=noise_obs,
                                        grav_const=beta,
                                        likelihoods=logl,
                                        nc_pred=nc_p,
                                        window_length=integration_window_length
                                    )

        save_dataframe_to_csv(pred_df, target_df_filename)
        print('Results saved to', target_df_filename)
        print('Inference finished.')
        

def run_inference(params):
    """
    Wrapper around `run_inference_trial_phase` using a parameter dictionary.

    Parameters
    ----------
    params : dict
        Must include keys:
        ['df_filename', 'target_df_filename', 'beta',
         'noise_KF', 'noise_obs', 'X', 'y', 'dt',
         'integration_window_length', 'preprocessing_function_for_data_reduction']

    Returns
    -------
    Any
        Output of `run_inference_trial_phase`.
    """
    return run_inference_trial_phase(
        df_filename=params['df_filename'],
        target_df_filename=params['target_df_filename'],
        beta=params['beta'],
        noise_KF=params['noise_KF'],
        noise_obs=params['noise_obs'],
        X=params['X'],
        y=params['y'],
        dt=params['dt'],
        integration_window_length=params['integration_window_length'],
        preprocessing_function_for_data_reduction=params['preprocessing_function_for_data_reduction']
    )



"""
MAGI FUNCTIONS
"""
def run_inference_trial_phase_MAGI(df_filename, target_df_filename, beta,
                                   noise_KF, noise_obs, X, y, dt,
                                   integration_window_length,
                                   preprocessing_function_for_data_reduction,
                                   reset_interval=1):
    """
    Run MAGI inference for all trial phases and save results to CSV.

    Parameters
    ----------
    df_filename : str
        Source dataset file path.
    target_df_filename : str
        Output CSV file for predictions.
    beta : float
        Gravity parameter used in model.
    noise_KF, noise_obs : float
        Noise parameters for filters.
    X : np.ndarray
        Training data (features).
    y : np.ndarray
        Training labels.
    dt : float
        Time step.
    integration_window_length : int
        Integration window size.
    preprocessing_function_for_data_reduction : callable
        Function applied to reduce glove data dimensionality.
    reset_interval : int
        Number of steps between belief resets (1 = no reset).
    """
    nc = models.NearestCentroidClassifier()
    nc.integration_window = integration_window_length
    nc.fit(X, y)

    if os.path.exists(target_df_filename):
        print(f"The file '{target_df_filename}' exists.")
    else:
        df = load_dataframe_from_csv(df_filename)
        pred_df = create_prediction_dataframe()

        for phase in range(1, df['block'].max() + 1):
            for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
                for rep_counter in range(df[(df['block'] == phase) &
                                            (df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                    df_att = df[(df['block'] == phase) &
                                (df['trial_number'] == trial_number) &
                                (df['rep_number'] == rep_counter)]

                    glove_data = df_att[glove_data_columns].to_numpy()

                    print('Running inference on:', target_df_filename)
                    print('Phase', phase, 'Trial', trial_number, 'Rep', rep_counter)

                    x0 = preprocessing_function_for_data_reduction(glove_data[0])
                    grav_model = models_jit.create_dirichlet_model(
                        X=X,
                        y=y,
                        noise_process=noise_KF,
                        noise_obs=noise_obs,
                        x0=x0[0],
                        dt=dt,
                        integration_window_length=integration_window_length,
                        gravitational_const=beta,
                        reset_interval=reset_interval,
                    )

                    obs = preprocessing_function_for_data_reduction(glove_data)
                    nc_pred = nc.run_sequence(obs)
                    means, var, ll = grav_model.run_sequence(observations=obs)

                    pred_df = add_rows_to_prediction_dataframe_batch(
                        df=pred_df,
                        phase=phase,
                        trial=trial_number,
                        repetition=rep_counter,
                        means=means,
                        vars=var,
                        noise_KF=noise_KF,
                        noise_obs=noise_obs,
                        grav_const=beta,
                        likelihoods=ll,
                        nc_preds=nc_pred,
                        window_length=integration_window_length)

        save_dataframe_to_csv(pred_df, target_df_filename)
        print('Results saved to', target_df_filename)
        print('Inference finished.')


def run_inference_trial_phase_MAGI_with_noise(df_filename, target_df_filename, beta,
                                              noise_KF, noise_obs, X, y, dt,
                                              integration_window_length,
                                              preprocessing_function_for_data_reduction,
                                              noise_lvl,
                                              reset_interval=1):
    """
    Run MAGI inference with additive noise for all trial phases.

    Parameters
    ----------
    <same as run_inference_trial_phase_MAGI, plus>
    noise_lvl : float
        Standard deviation of additive Gaussian noise.
    reset_interval : int
        Number of steps between belief resets (1 = no reset).
    """
    nc = models.NearestCentroidClassifier()
    nc.integration_window = integration_window_length
    nc.fit(X, y)

    if os.path.exists(target_df_filename):
        print(f"The file '{target_df_filename}' exists.")
    else:
        df = load_dataframe_from_csv(df_filename)
        pred_df = create_prediction_dataframe()

        for phase in range(1, df['block'].max() + 1):
            for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
                for rep_counter in range(df[(df['block'] == phase) &
                                            (df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                    df_att = df[(df['block'] == phase) &
                                (df['trial_number'] == trial_number) &
                                (df['rep_number'] == rep_counter)]

                    glove_data = df_att[glove_data_columns].to_numpy()
                    noise = np.random.normal(loc=0.0, scale=noise_lvl, size=glove_data.shape)
                    glove_data = glove_data + noise

                    print('Running inference on:', target_df_filename)
                    print('Phase', phase, 'Trial', trial_number, 'Rep', rep_counter)

                    x0 = preprocessing_function_for_data_reduction(glove_data[0])
                    grav_model = models_jit.create_dirichlet_model(
                        X=X,
                        y=y,
                        noise_process=noise_KF,
                        noise_obs=noise_obs,
                        x0=x0[0],
                        dt=dt,
                        integration_window_length=integration_window_length,
                        gravitational_const=beta,
                        reset_interval=reset_interval,
                    )

                    obs = preprocessing_function_for_data_reduction(glove_data)
                    nc_pred = nc.run_sequence(obs)
                    means, var, ll = grav_model.run_sequence(observations=obs)

                    pred_df = add_rows_to_prediction_dataframe_batch(
                        df=pred_df,
                        phase=phase,
                        trial=trial_number,
                        repetition=rep_counter,
                        means=means,
                        vars=var,
                        noise_KF=noise_KF,
                        noise_obs=noise_obs,
                        grav_const=beta,
                        likelihoods=ll,
                        nc_preds=nc_pred,
                        window_length=integration_window_length)

        save_dataframe_to_csv(pred_df, target_df_filename)
        print('Results saved to', target_df_filename)
        print('Inference finished.')


def run_inference_MAGI(params):
    """Wrapper that unpacks params dict and calls run_inference_trial_phase_MAGI."""
    return run_inference_trial_phase_MAGI(
        df_filename=params['df_filename'],
        target_df_filename=params['target_df_filename'],
        beta=params['beta'],
        noise_KF=params['noise_KF'],
        noise_obs=params['noise_obs'],
        X=params['X'],
        y=params['y'],
        dt=params['dt'],
        integration_window_length=params['integration_window_length'],
        preprocessing_function_for_data_reduction=params['preprocessing_function_for_data_reduction'],
        reset_interval=params.get('reset_interval', 1),   # <-- NEW
    )




def run_inference_MAGI_with_noise(params):
    """Wrapper that unpacks params dict and calls run_inference_trial_phase_MAGI_with_noise."""
    return run_inference_trial_phase_MAGI_with_noise(
        df_filename=params['df_filename'],
        target_df_filename=params['target_df_filename'],
        beta=params['beta'],
        noise_KF=params['noise_KF'],
        noise_obs=params['noise_obs'],
        X=params['X'],
        y=params['y'],
        dt=params['dt'],
        integration_window_length=params['integration_window_length'],
        preprocessing_function_for_data_reduction=params['preprocessing_function_for_data_reduction'],
        noise_lvl=params['noise_lvl'],
        reset_interval=params.get('reset_interval', 1),   # <-- NEW
    )



def process_participant(data_frame, pred_df):
    """
    Process one participant and save their negative latency times data.
    
    Analyzes prediction timing relative to object lift events to calculate
    negative latency (i.e., how early predictions occur before physical actions).
    
    Args:
        args: Tuple of (name_counter, name) where name_counter is participant index
              and name is participant identifier string
    
    Returns:
        String summary of processing results
    """
    
    # Initialize list to store timing data for all trials
    all_times = []
    df = data_frame
    mean_data_columns = [f'mean_pred_{i}' for i in range(9)]  # Mean predictions for 9 grasp classes
    var_data_columns = [f'var_pred_{i}' for i in range(9)]  # Variance predictions for 9 grasp classes
    nc_pred_data_columns = [f'nc_pred_{i}' for i in range(9)]  # Non-causal predictions for 9 grasp classes
    for phase in np.arange(1, pred_df['block'].max() + 1):
        # Loop through trials within each block
        # for trial_number in np.arange(pred_df[pred_df['block'] == phase]['trial_number'].max() + 1):
        for trial_number in pred_df[pred_df['block'] == phase]["trial_number"].unique():
            # Loop through repetitions within each trial
            for rep_counter in np.arange(pred_df[(pred_df['block'] == phase) & (pred_df['trial_number'] == trial_number)]['rep_number'].max() + 1):
                
                # === Data selection ===
                # Filter prediction data for current trial
                pred_df_att = pred_df[(pred_df['block'] == phase) &
                                      (pred_df['trial_number'] == trial_number) &
                                      (pred_df['rep_number'] == rep_counter)]
                
                # Filter raw data for current trial (only successful trials)
                df_att = df[(df['block'] == phase) &
                            (df['trial_number'] == trial_number) &
                            (df['rep_number'] == rep_counter) &
                            (df['trial_success'] == 1)]

                # Re-filter data (redundant, but kept as in original)
                pred_df_att = pred_df[(pred_df['block'] == phase) & (pred_df['trial_number'] == trial_number) & (pred_df['rep_number'] == rep_counter)]         
                df_att = df[(df['block'] == phase) & (df['trial_number'] == trial_number) & (df['rep_number'] == rep_counter) & (df['trial_success'] == 1)]    
                
                # Extract glove sensor data and timestamps
                glove_time = df_att['glove_timestamps'].to_numpy()
                glove_data = df_att[glove_data_columns].to_numpy()
                
                # Extract force sensor board data and clean outliers
                board_data = df_att[board_data_columns].to_numpy()
                board_data = replace_with_highest(arr=board_data, target=8191.0)  # Replace sensor saturation value
                board_data = replace_with_highest(arr=board_data, target=8190.0)  # Replace near-saturation value
                board_time = df_att['board_timestamps'].to_numpy()
                
                # Extract palm sensor data, clean outliers, and normalize to start at zero
                palm_data = df_att['palm_data'].to_numpy()
                palm_data = replace_with_highest(arr=palm_data, target=8191.0)
                palm_data = replace_with_highest(arr=palm_data, target=8190.0)
                palm_data = palm_data-palm_data[0]  # Zero-baseline the palm data
                palm_time = df_att['palm_timestamps'].to_numpy()
                
                # Get true target object that was grasped
                y_true = int(df_att['target_object'].to_numpy()[-1])
                
                # Set analysis window start point
                start = 0
        
                # Get lifted target object (same as y_true, redundant)
                lifted_target = int(df_att['target_object'].to_numpy()[-1])
                
                # Extract prediction arrays
                means = pred_df_att[mean_data_columns].to_numpy()  # Mean predictions
                var = pred_df_att[var_data_columns].to_numpy()  # Variance predictions
                nc_pred = pred_df_att[nc_pred_data_columns].to_numpy()  # Non-causal predictions
                
                # Compute movement speed profile from glove data
                speed = transform_data_to_speed_profile(glove_data, glove_time)
                
                # === Stability analysis ===
                # Analyze prediction stability during grasp execution
                end = -1  # Use all data up to last point
                
                # Extract board sensor time series for current trial and normalize
                board_ts = board_data[start:end, int(trial_number)]
                board_ts = board_ts - board_ts[0]
                
                # Find object lift-off time (10% of max force threshold)
                t_lift = first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)
                
                # Find object put-down time by analyzing time series in reverse
                board_ts_flip = board_ts[::-1]-board_ts[end]
                t_down = len(board_ts)-first_change_index(time_series=board_ts_flip, threshold=board_ts.max()*0.1)
                if t_down <= t_lift: t_down = len(board_ts)-1  # Ensure t_down is after t_lift
                
                # Get predictions during grasp (from lift to put-down)
                pred_dur_grasp = means.argmax(axis=1)[t_lift:t_down]  # Causal model predictions
                nc_pred_dur_grasp = nc_pred.argmax(axis=1)[t_lift:t_down]  # Non-causal predictions
                
                # Find hand lift-off time from palm sensor (10% threshold)
                hand_lift = first_change_index(time_series=palm_data, threshold=palm_data.max()*0.1)
 
                # Calculate prediction stability: proportion of correct predictions during grasp
                stability_pred = np.count_nonzero(pred_dur_grasp == y_true)/pred_dur_grasp.shape[0]
                stability_nc = np.count_nonzero(nc_pred_dur_grasp == y_true)/nc_pred_dur_grasp.shape[0]
                
                # === Negative latency timing analysis ===
                # Find exact moment of maximum force (object lift peak)
                end = np.argmax(board_data[:,int(trial_number)])
                
                # Get moment of object lift-off
                board_ts = board_data[start:end, int(trial_number)]
                board_ts = board_ts - board_ts[0]
                t_lift = first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)
                
                # Get moment of non-causal prediction stabilization
                t_nc_pred = find_first_of_final_sequence(nc_pred[start:end].argmax(axis=1), nc_pred[end].argmax())
                
                # === Get moment of gravitational model prediction (method 1: single class) ===
                # Find when the true class is first predicted and remains stable until lift
                incrementer = 0
                t_grav_pred_1 = end
                while t_lift+incrementer < end:
                    t_grav_pred_1 = find_first_of_final_sequence(means.argmax(axis=1)[start:t_lift+incrementer], y_true) + start
                    # Check if prediction is correct at this timepoint
                    if means[start+t_lift+incrementer].argmax() == y_true: break
                    else: incrementer = incrementer + 1
                
                # Verify prediction consistency with non-causal model
                if means.argmax(axis=1)[start+t_nc_pred]==nc_pred[end].argmax(): t_grav_pred_1=t_grav_pred_1
                else: t_grav_pred=t_nc_pred+start
                
                # Analyze probability distribution to get top-k predictions
                cumsum, sets = analyze_probability_distribution(means)
                
                # === Get moment of gravitational model prediction (method 2: top-2 classes) ===
                t_grav_pred_2 = find_first_of_final_sequence(sets[start:start+end,:2], y_true) + start
                
                # === Get moment of gravitational model prediction (method 3: top-3 classes) ===
                t_grav_pred_3 = find_first_of_final_sequence(sets[start:start+end,:3], y_true) + start
                
                # === Switching times analysis ===
                # Analyze how often predictions switch between classes
                switching_list = []
                for k in range(1,8):  # For top-1 through top-7 predictions
                    # Find indices where prediction switches
                    switch_moments = find_switch_indices(sets[start:t_down,0:k])
                    # Calculate duration of each prediction segment
                    switch_length = np.diff(switch_moments)
                    # Check accuracy of each prediction segment
                    switch_acc_lst = []
                    for l in range(len(switch_moments)-1):                        
                        switch = switch_moments[l]
                        if y_true in sets[start+switch,0:k]: switch_acc_lst.append(1)
                        else: switch_acc_lst.append(0)
                    switching_list.append(switch_length)
                    switching_list.append(switch_acc_lst)
                
                # === Calculate distance traveled analysis ===
                # Initialize timing results dictionary
                times_dict = {}
                cl = ['black', 'blue', 'green'][::-1]  # Colors for visualization (not used here)
                
                # Calculate hand movement distance up to lift
                mov_data = preprocess_data_reduction(glove_data[start:start+t_lift], 5)
                distance_travelled_lift = compute_distance_travelled_in_percent(mov_data, target=mov_data[-1])
                
                # Calculate hand movement distance up to non-causal prediction time
                try: 
                    mov_data = preprocess_data_reduction(glove_data[start:start+t_nc_pred], 5)
                    distance_travelled_nc_pred = compute_distance_travelled_in_percent(mov_data, target=mov_data[-1])
                except: 
                    # If calculation fails, assume full distance traveled
                    distance_travelled_nc_pred = [100]*len(mov_data)
                
                # === Store results in dictionary ===
                permutations = perms  # Not used in this section
                times_dict = {}
                
                # Check if causal model ever predicts correctly
                # print(means)
                if y_true in means[t_lift:t_down].argmax(axis=1): times_dict['correct'] = 1
                else : times_dict['correct'] = 0
                
                # Check if non-causal model ever predicts correctly
                if y_true in nc_pred[t_lift:t_down].argmax(axis=1): times_dict['nc_correct'] = 1
                else : times_dict['nc_correct'] = 0

                # Store experimental metadata
                times_dict['block'] = phase  # Experimental block number
                times_dict['trial_number'] = trial_number  # Trial number
                times_dict['rep_number'] = rep_counter  # Repetition number
                times_dict['movement_start'] = glove_time[np.where(df_att['hand_active'] == 1)[0][0]]  # Movement onset time
                times_dict['lifted_target'] = lifted_target  # Object that was grasped
                times_dict['t_lift'] = glove_time[start:end][t_lift]  # Object lift-off timestamp
                times_dict['t_hand'] = glove_time[start:end][hand_lift]  # Hand lift-off timestamp
                
                # Store switching analysis results
                times_dict['switching_list'] = list(switching_list)
                
                # Store prediction stability metrics
                times_dict['stability_pred'] = stability_pred  # Causal model stability
                times_dict['stability_nc'] = stability_nc  # Non-causal model stability
                
                # Store key prediction timestamps
                times_dict['t_nc'] = glove_time[start:end][t_nc_pred]  # Non-causal prediction time
                times_dict['t_nl1'] = glove_time[t_grav_pred_1]  # Top-1 prediction time
                times_dict['t_nl2'] = glove_time[t_grav_pred_2]  # Top-2 prediction time
                times_dict['t_nl3'] = glove_time[t_grav_pred_3]  # Top-3 prediction time

                # Calculate negative latency relative to non-causal prediction
                # Negative latency = how much earlier the causal model predicts vs. baseline
                times_dict['negative_latency_k1_nc'] = max(glove_time[hand_lift]-glove_time[start:end][t_nc_pred], glove_time[t_grav_pred_1] - glove_time[start:end][t_nc_pred])
                times_dict['negative_latency_k2_nc'] = max(glove_time[hand_lift]-glove_time[start:end][t_nc_pred], glove_time[t_grav_pred_2] - glove_time[start:end][t_nc_pred])
                times_dict['negative_latency_k3_nc'] = max(glove_time[hand_lift]-glove_time[start:end][t_nc_pred], glove_time[t_grav_pred_3] - glove_time[start:end][t_nc_pred])
                
                # Calculate negative latency relative to object lift
                times_dict['negative_latency_k1_lift'] = max(glove_time[hand_lift]-glove_time[start:end][t_lift], glove_time[t_grav_pred_1] - glove_time[start:end][t_lift])
                times_dict['negative_latency_k2_lift'] = max(glove_time[hand_lift]-glove_time[start:end][t_lift], glove_time[t_grav_pred_2] - glove_time[start:end][t_lift])
                times_dict['negative_latency_k3_lift'] = max(glove_time[hand_lift]-glove_time[start:end][t_lift], glove_time[t_grav_pred_3] - glove_time[start:end][t_lift])
                
                # Store target and predicted objects
                times_dict['lifted_target'] = lifted_target  # True grasped object
                times_dict['predicted_target_1'] = sets[start+t_grav_pred_1,:1]  # Top-1 predicted object
                times_dict['predicted_target_2'] = list(sets[start+t_grav_pred_2,:2])  # Top-2 predicted objects
                times_dict['predicted_target_3'] = list(sets[start+t_grav_pred_3,:3])  # Top-3 predicted objects
                
                # Store complete time series
                times_dict['times'] = glove_time[start:t_lift]
                
                # Store percentage of movement completed at each prediction time (relative to lift)
                if t_grav_pred_1 < len(distance_travelled_lift):
                    times_dict['travelled_dist_lift_in_percent_nl1'] = distance_travelled_lift[t_grav_pred_1]
                else: times_dict['travelled_dist_lift_in_percent_nl1'] = np.nan
                
                if t_grav_pred_2 < len(distance_travelled_lift):
                    times_dict['travelled_dist_lift_in_percent_nl2'] = distance_travelled_lift[t_grav_pred_2]
                else: times_dict['travelled_dist_lift_in_percent_nl2'] = np.nan
                
                if t_grav_pred_3< len(distance_travelled_lift):
                    times_dict['travelled_dist_lift_in_percent_nl3'] = distance_travelled_lift[t_grav_pred_3]
                else: times_dict['travelled_dist_lift_in_percent_nl3'] = np.nan
                
                # Store percentage of movement completed at each prediction time (relative to NC prediction)
                if t_grav_pred_1 < len(distance_travelled_nc_pred):
                    times_dict['travelled_dist_nc_in_percent_nl1'] = distance_travelled_nc_pred[t_grav_pred_1]
                else: times_dict['travelled_dist_nc_in_percent_nl1'] = np.nan
                
                if t_grav_pred_2 < len(distance_travelled_nc_pred):
                    times_dict['travelled_dist_nc_in_percent_nl2'] = distance_travelled_nc_pred[t_grav_pred_2]
                else: times_dict['travelled_dist_nc_in_percent_nl2'] = np.nan
                
                if t_grav_pred_3< len(distance_travelled_nc_pred):
                    times_dict['travelled_dist_nc_in_percent_nl3'] = distance_travelled_nc_pred[t_grav_pred_3]
                else: times_dict['travelled_dist_nc_in_percent_nl3'] = np.nan
                
                # Store complete distance traveled profile
                times_dict['travelled_dist_lift_in_percent'] = distance_travelled_lift
                
                # Append trial results to list
                all_times.append(times_dict)
                # print(glove_time[t_grav_pred_1] )
                
    # Save all timing results to CSV file
    df_times = pd.DataFrame(all_times)
    return df_times

def process_participant_cross_val(data_frame, pred_df):
    """
    Process one participant and save their negative latency times data.
    
    Analyzes prediction timing relative to object lift events to calculate
    negative latency (i.e., how early predictions occur before physical actions).
    
    Args:
        args: Tuple of (name_counter, name) where name_counter is participant index
              and name is participant identifier string
    
    Returns:
        String summary of processing results
    """
    # Initialize list to store timing data for all trials
    all_times = []
    df = data_frame
    mean_data_columns = [f'mean_pred_{i}' for i in range(9)]  # Mean predictions for 9 grasp classes
    var_data_columns = [f'var_pred_{i}' for i in range(9)]  # Variance predictions for 9 grasp classes
    nc_pred_data_columns = [f'nc_pred_{i}' for i in range(9)]  # Non-causal predictions for 9 grasp classes
    for phase in pred_df['block'].unique():
        # Loop through trials within each block
        # for trial_number in np.arange(pred_df[pred_df['block'] == phase]['trial_number'].max() + 1):
        for trial_number in pred_df[pred_df['block'] == phase]["trial_number"].unique():
            # Loop through repetitions within each trial
            for rep_counter in np.arange(pred_df[(pred_df['block'] == phase) & (pred_df['trial_number'] == trial_number)]['rep_number'].max() + 1):
                
                # === Data selection ===
                # Filter prediction data for current trial
                pred_df_att = pred_df[(pred_df['block'] == phase) &
                                      (pred_df['trial_number'] == trial_number) &
                                      (pred_df['rep_number'] == rep_counter)]
                
                # Filter raw data for current trial (only successful trials)
                df_att = df[(df['block'] == phase) &
                            (df['trial_number'] == trial_number) &
                            (df['rep_number'] == rep_counter) &
                            (df['trial_success'] == 1)]

                # Re-filter data (redundant, but kept as in original)
                pred_df_att = pred_df[(pred_df['block'] == phase) & (pred_df['trial_number'] == trial_number) & (pred_df['rep_number'] == rep_counter)]         
                df_att = df[(df['block'] == phase) & (df['trial_number'] == trial_number) & (df['rep_number'] == rep_counter) & (df['trial_success'] == 1)]    
                
                # Extract glove sensor data and timestamps
                glove_time = df_att['glove_timestamps'].to_numpy()
                glove_data = df_att[glove_data_columns].to_numpy()
                
                # Extract force sensor board data and clean outliers
                board_data = df_att[board_data_columns].to_numpy()
                board_data = replace_with_highest(arr=board_data, target=8191.0)  # Replace sensor saturation value
                board_data = replace_with_highest(arr=board_data, target=8190.0)  # Replace near-saturation value
                board_time = df_att['board_timestamps'].to_numpy()
                
                # Extract palm sensor data, clean outliers, and normalize to start at zero
                palm_data = df_att['palm_data'].to_numpy()
                palm_data = replace_with_highest(arr=palm_data, target=8191.0)
                palm_data = replace_with_highest(arr=palm_data, target=8190.0)
                palm_data = palm_data-palm_data[0]  # Zero-baseline the palm data
                palm_time = df_att['palm_timestamps'].to_numpy()
                
                # Get true target object that was grasped
                y_true = int(df_att['target_object'].to_numpy()[-1])
                
                # Set analysis window start point
                start = 0
        
                # Get lifted target object (same as y_true, redundant)
                lifted_target = int(df_att['target_object'].to_numpy()[-1])
                
                # Extract prediction arrays
                means = pred_df_att[mean_data_columns].to_numpy()  # Mean predictions
                var = pred_df_att[var_data_columns].to_numpy()  # Variance predictions
                nc_pred = pred_df_att[nc_pred_data_columns].to_numpy()  # Non-causal predictions
                
                # Compute movement speed profile from glove data
                speed = transform_data_to_speed_profile(glove_data, glove_time)
                
                # === Stability analysis ===
                # Analyze prediction stability during grasp execution
                end = -1  # Use all data up to last point
                
                # Extract board sensor time series for current trial and normalize
                board_ts = board_data[start:end, int(trial_number)]
                board_ts = board_ts - board_ts[0]
                
                # Find object lift-off time (10% of max force threshold)
                t_lift = first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)
                
                # Find object put-down time by analyzing time series in reverse
                board_ts_flip = board_ts[::-1]-board_ts[end]
                t_down = len(board_ts)-first_change_index(time_series=board_ts_flip, threshold=board_ts.max()*0.1)
                if t_down <= t_lift: t_down = len(board_ts)-1  # Ensure t_down is after t_lift
                
                # Get predictions during grasp (from lift to put-down)
                pred_dur_grasp = means.argmax(axis=1)[t_lift:t_down]  # Causal model predictions
                nc_pred_dur_grasp = nc_pred.argmax(axis=1)[t_lift:t_down]  # Non-causal predictions
                
                # Find hand lift-off time from palm sensor (10% threshold)
                hand_lift = first_change_index(time_series=palm_data, threshold=palm_data.max()*0.1)
 
                # Calculate prediction stability: proportion of correct predictions during grasp
                stability_pred = np.count_nonzero(pred_dur_grasp == y_true)/pred_dur_grasp.shape[0]
                stability_nc = np.count_nonzero(nc_pred_dur_grasp == y_true)/nc_pred_dur_grasp.shape[0]
                
                # === Negative latency timing analysis ===
                # Find exact moment of maximum force (object lift peak)
                end = np.argmax(board_data[:,int(trial_number)])
                
                # Get moment of object lift-off
                board_ts = board_data[start:end, int(trial_number)]
                board_ts = board_ts - board_ts[0]
                t_lift = first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)
                
                # Get moment of non-causal prediction stabilization
                t_nc_pred = find_first_of_final_sequence(nc_pred[start:end].argmax(axis=1), nc_pred[end].argmax())
                
                # === Get moment of gravitational model prediction (method 1: single class) ===
                # Find when the true class is first predicted and remains stable until lift
                incrementer = 0
                t_grav_pred_1 = end
                while t_lift+incrementer < end:
                    t_grav_pred_1 = find_first_of_final_sequence(means.argmax(axis=1)[start:t_lift+incrementer], y_true) + start
                    # Check if prediction is correct at this timepoint
                    if means[start+t_lift+incrementer].argmax() == y_true: break
                    else: incrementer = incrementer + 1
                
                # Verify prediction consistency with non-causal model
                if means.argmax(axis=1)[start+t_nc_pred]==nc_pred[end].argmax(): t_grav_pred_1=t_grav_pred_1
                else: t_grav_pred=t_nc_pred+start
                
                # Analyze probability distribution to get top-k predictions
                cumsum, sets = analyze_probability_distribution(means)
                
                # === Get moment of gravitational model prediction (method 2: top-2 classes) ===
                t_grav_pred_2 = find_first_of_final_sequence(sets[start:start+end,:2], y_true) + start
                
                # === Get moment of gravitational model prediction (method 3: top-3 classes) ===
                t_grav_pred_3 = find_first_of_final_sequence(sets[start:start+end,:3], y_true) + start
                
                # === Switching times analysis ===
                # Analyze how often predictions switch between classes
                switching_list = []
                for k in range(1,8):  # For top-1 through top-7 predictions
                    # Find indices where prediction switches
                    switch_moments = find_switch_indices(sets[start:t_down,0:k])
                    # Calculate duration of each prediction segment
                    switch_length = np.diff(switch_moments)
                    # Check accuracy of each prediction segment
                    switch_acc_lst = []
                    for l in range(len(switch_moments)-1):                        
                        switch = switch_moments[l]
                        if y_true in sets[start+switch,0:k]: switch_acc_lst.append(1)
                        else: switch_acc_lst.append(0)
                    switching_list.append(switch_length)
                    switching_list.append(switch_acc_lst)
                
                # === Calculate distance traveled analysis ===
                # Initialize timing results dictionary
                times_dict = {}
                cl = ['black', 'blue', 'green'][::-1]  # Colors for visualization (not used here)
                
                # Calculate hand movement distance up to lift
                mov_data = preprocess_data_reduction(glove_data[start:start+t_lift], 5)
                distance_travelled_lift = compute_distance_travelled_in_percent(mov_data, target=mov_data[-1])
                
                # Calculate hand movement distance up to non-causal prediction time
                try: 
                    mov_data = preprocess_data_reduction(glove_data[start:start+t_nc_pred], 5)
                    distance_travelled_nc_pred = compute_distance_travelled_in_percent(mov_data, target=mov_data[-1])
                except: 
                    # If calculation fails, assume full distance traveled
                    distance_travelled_nc_pred = [100]*len(mov_data)
                
                # === Store results in dictionary ===
                permutations = perms  # Not used in this section
                times_dict = {}
                
                # Check if causal model ever predicts correctly
                # print(means)
                if y_true in means[t_lift:t_down].argmax(axis=1): times_dict['correct'] = 1
                else : times_dict['correct'] = 0
                
                # Check if non-causal model ever predicts correctly
                if y_true in nc_pred[t_lift:t_down].argmax(axis=1): times_dict['nc_correct'] = 1
                else : times_dict['nc_correct'] = 0

                # Store experimental metadata
                times_dict['block'] = phase  # Experimental block number
                times_dict['trial_number'] = trial_number  # Trial number
                times_dict['rep_number'] = rep_counter  # Repetition number
                times_dict['movement_start'] = glove_time[np.where(df_att['hand_active'] == 1)[0][0]]  # Movement onset time
                times_dict['lifted_target'] = lifted_target  # Object that was grasped
                times_dict['t_lift'] = glove_time[start:end][t_lift]  # Object lift-off timestamp
                times_dict['t_hand'] = glove_time[start:end][hand_lift]  # Hand lift-off timestamp
                
                # Store switching analysis results
                times_dict['switching_list'] = list(switching_list)
                
                # Store prediction stability metrics
                times_dict['stability_pred'] = stability_pred  # Causal model stability
                times_dict['stability_nc'] = stability_nc  # Non-causal model stability
                
                # Store key prediction timestamps
                times_dict['t_nc'] = glove_time[start:end][t_nc_pred]  # Non-causal prediction time
                times_dict['t_nl1'] = glove_time[t_grav_pred_1]  # Top-1 prediction time
                times_dict['t_nl2'] = glove_time[t_grav_pred_2]  # Top-2 prediction time
                times_dict['t_nl3'] = glove_time[t_grav_pred_3]  # Top-3 prediction time

                # Calculate negative latency relative to non-causal prediction
                # Negative latency = how much earlier the causal model predicts vs. baseline
                times_dict['negative_latency_k1_nc'] = max(glove_time[hand_lift]-glove_time[start:end][t_nc_pred], glove_time[t_grav_pred_1] - glove_time[start:end][t_nc_pred])
                times_dict['negative_latency_k2_nc'] = max(glove_time[hand_lift]-glove_time[start:end][t_nc_pred], glove_time[t_grav_pred_2] - glove_time[start:end][t_nc_pred])
                times_dict['negative_latency_k3_nc'] = max(glove_time[hand_lift]-glove_time[start:end][t_nc_pred], glove_time[t_grav_pred_3] - glove_time[start:end][t_nc_pred])
                
                # Calculate negative latency relative to object lift
                times_dict['negative_latency_k1_lift'] = max(glove_time[hand_lift]-glove_time[start:end][t_lift], glove_time[t_grav_pred_1] - glove_time[start:end][t_lift])
                times_dict['negative_latency_k2_lift'] = max(glove_time[hand_lift]-glove_time[start:end][t_lift], glove_time[t_grav_pred_2] - glove_time[start:end][t_lift])
                times_dict['negative_latency_k3_lift'] = max(glove_time[hand_lift]-glove_time[start:end][t_lift], glove_time[t_grav_pred_3] - glove_time[start:end][t_lift])
                
                # Store target and predicted objects
                times_dict['lifted_target'] = lifted_target  # True grasped object
                times_dict['predicted_target_1'] = sets[start+t_grav_pred_1,:1]  # Top-1 predicted object
                times_dict['predicted_target_2'] = list(sets[start+t_grav_pred_2,:2])  # Top-2 predicted objects
                times_dict['predicted_target_3'] = list(sets[start+t_grav_pred_3,:3])  # Top-3 predicted objects
                
                # Store complete time series
                times_dict['times'] = glove_time[start:t_lift]
                
                # Store percentage of movement completed at each prediction time (relative to lift)
                if t_grav_pred_1 < len(distance_travelled_lift):
                    times_dict['travelled_dist_lift_in_percent_nl1'] = distance_travelled_lift[t_grav_pred_1]
                else: times_dict['travelled_dist_lift_in_percent_nl1'] = np.nan
                
                if t_grav_pred_2 < len(distance_travelled_lift):
                    times_dict['travelled_dist_lift_in_percent_nl2'] = distance_travelled_lift[t_grav_pred_2]
                else: times_dict['travelled_dist_lift_in_percent_nl2'] = np.nan
                
                if t_grav_pred_3< len(distance_travelled_lift):
                    times_dict['travelled_dist_lift_in_percent_nl3'] = distance_travelled_lift[t_grav_pred_3]
                else: times_dict['travelled_dist_lift_in_percent_nl3'] = np.nan
                
                # Store percentage of movement completed at each prediction time (relative to NC prediction)
                if t_grav_pred_1 < len(distance_travelled_nc_pred):
                    times_dict['travelled_dist_nc_in_percent_nl1'] = distance_travelled_nc_pred[t_grav_pred_1]
                else: times_dict['travelled_dist_nc_in_percent_nl1'] = np.nan
                
                if t_grav_pred_2 < len(distance_travelled_nc_pred):
                    times_dict['travelled_dist_nc_in_percent_nl2'] = distance_travelled_nc_pred[t_grav_pred_2]
                else: times_dict['travelled_dist_nc_in_percent_nl2'] = np.nan
                
                if t_grav_pred_3< len(distance_travelled_nc_pred):
                    times_dict['travelled_dist_nc_in_percent_nl3'] = distance_travelled_nc_pred[t_grav_pred_3]
                else: times_dict['travelled_dist_nc_in_percent_nl3'] = np.nan
                
                # Store complete distance traveled profile
                times_dict['travelled_dist_lift_in_percent'] = distance_travelled_lift
                
                # Append trial results to list
                all_times.append(times_dict)
                # print(glove_time[t_grav_pred_1] )
                
    # Save all timing results to CSV file
    df_times = pd.DataFrame(all_times)
    return df_times
from pathlib import Path
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DATA_ROOT = Path("/Users/arthu/Desktop/CTM_Res-Pro/data/reach_to_grasp_dataset/data")


FINGERTIP_SENSOR_IDS = [4, 9, 14, 19, 24]

FINGERTIP_COLUMNS = [
    f"SensorID_{sensor_id}_{axis}"
    for sensor_id in FINGERTIP_SENSOR_IDS
    for axis in ["x", "y", "z"]
]

ALL_GLOVE_COLUMNS = [
    f"SensorID_{sensor_id}_{axis}"
    for sensor_id in range(25)
    for axis in ["x", "y", "z"]
]

N_TIMEPOINTS = 100

# ---------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------

TRIAL_PATTERN = re.compile(
    r"B(?P<block>\d+)"
    r"_trial(?P<trial>\d+)"
    r"_rep(?P<rep>\d+)"
    r"_attempt(?P<attempt>\d+)"
    r"\.csv$"
)


def parse_trial_filename(path):
    """
    Extract experimental metadata from a trial filename.

    Example:
    B0_trial0_rep1_attempt0.csv

    becomes:
    block=0, trial=0, rep=1, attempt=0
    """

    path = Path(path)

    match = TRIAL_PATTERN.match(path.name)

    if match is None:
        raise ValueError(f"Unexpected filename format: {path.name}")

    return {
        "block": int(match.group("block")),
        "trial": int(match.group("trial")),
        "rep": int(match.group("rep")),
        "attempt": int(match.group("attempt")),
    }


# ---------------------------------------------------------------------
# Recording inventory
# ---------------------------------------------------------------------


def build_recording_inventory(data_root=DATA_ROOT):
    """
    Build an inventory of all processed per-trial recordings.

    This function only records what files exist and what their filenames
    encode. It does not apply label corrections, failure filtering,
    or movement extraction.
    """

    records = []

    for participant_dir in sorted(data_root.glob("participant_*")):
        participant_match = re.fullmatch(
            r"participant_(\d+)",
            participant_dir.name,
        )

        if participant_match is None:
            continue

        participant_id = int(participant_match.group(1))

        motion_dir = participant_dir / "motion"

        if not motion_dir.exists():
            continue

        files = sorted(
            path
            for path in motion_dir.glob("*.csv")
            if not path.name.endswith("_raw.csv")
        )

        for path in files:
            info = parse_trial_filename(path)

            records.append(
                {
                    "participant": participant_id,
                    "block": info["block"],
                    "trial_number": info["trial"],
                    "rep_number": info["rep"],
                    "attempt_number": info["attempt"],
                    "file": path.name,
                    "path": str(path),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Final valid metadata value
# ---------------------------------------------------------------------


def final_valid_value(series, field_name):
    """
    Return the final non-missing value from a metadata column.
    """

    valid = series.dropna()

    if valid.empty:
        raise ValueError(f"No valid value found for '{field_name}'.")

    return valid.iloc[-1]


# ---------------------------------------------------------------------
# Raw metadata extraction
# ---------------------------------------------------------------------


def extract_raw_metadata(path):
    """
    Extract raw trial-level metadata from one processed recording.

    No experimental corrections or derived labels are applied here.
    """

    path = Path(path)

    participant_match = re.search(
        r"participant_(\d+)",
        str(path),
    )

    if participant_match is None:
        raise ValueError(f"Could not determine participant from path: {path}")

    df = pd.read_csv(
        path,
        usecols=[
            "block",
            "trial_number",
            "rep_number",
            "attempt_number",
            "target_object",
            "trial_success",
        ],
    )

    target_values = df["target_object"].dropna().unique().tolist()

    metadata = {
        "participant": int(participant_match.group(1)),
        "block": int(
            final_valid_value(
                df["block"],
                "block",
            )
        ),
        "trial_number": int(
            final_valid_value(
                df["trial_number"],
                "trial_number",
            )
        ),
        "rep_number": int(
            final_valid_value(
                df["rep_number"],
                "rep_number",
            )
        ),
        "attempt_number": int(
            final_valid_value(
                df["attempt_number"],
                "attempt_number",
            )
        ),
        "trial_success": int(
            final_valid_value(
                df["trial_success"],
                "trial_success",
            )
        ),
        "target_object_final": int(
            final_valid_value(
                df["target_object"],
                "target_object",
            )
        ),
        "target_values_seen": target_values,
        "target_changed_within_recording": (len(target_values) > 1),
        "file": path.name,
        "path": str(path),
    }

    return metadata


# ------------------------------------------------------------
# Selecting Successful Recording
# ------------------------------------------------------------


def select_successful_recordings(metadata_df):
    """
    Select the successful recording for each logical trial slot.

    A logical slot is defined by:
        participant, block, trial_number, rep_number

    Exactly one successful attempt is expected for retained slots.
    Slots with no successful attempt are returned separately.
    """

    slot_columns = [
        "participant",
        "block",
        "trial_number",
        "rep_number",
    ]

    selected_records = []
    excluded_slots = []

    for slot_key, group in metadata_df.groupby(
        slot_columns,
        sort=True,
    ):
        successful = group[group["trial_success"] == 1]

        if len(successful) > 1:
            raise ValueError(
                f"More than one successful attempt found for slot={slot_key}"
            )

        if len(successful) == 1:
            selected_records.append(successful.iloc[0].to_dict())

        else:
            excluded_slots.append(
                {
                    "participant": slot_key[0],
                    "block": slot_key[1],
                    "trial_number": slot_key[2],
                    "rep_number": slot_key[3],
                    "n_attempts": len(group),
                    "files": group["file"].tolist(),
                }
            )

    selected_df = pd.DataFrame(selected_records)

    excluded_df = pd.DataFrame(excluded_slots)

    return selected_df, excluded_df


# ------------------------------------------------------------
# Applying Target Correction for: rep0
# ------------------------------------------------------------


def apply_rep0_target_correction(
    metadata,
    selected_metadata,
):
    """
    Apply the B1-B4 repetition-0 target correction.

    For repetition 0, use the final valid target from the
    selected successful repetition 1 recording belonging to
    the same participant, block, and trial.

    The source CSV is never modified.
    """

    corrected = metadata.copy()

    corrected["target_object_raw"] = metadata["target_object_final"]

    corrected["target_object_corrected"] = metadata["target_object_final"]

    corrected["target_correction_applied"] = False
    corrected["target_reference_rep"] = None
    corrected["target_reference_file"] = None

    # B0 does not use this correction
    if metadata["block"] == 0:
        return corrected

    # Only repetition 0 is corrected
    if metadata["rep_number"] != 0:
        return corrected

    reference = selected_metadata[
        (selected_metadata["participant"] == metadata["participant"])
        & (selected_metadata["block"] == metadata["block"])
        & (selected_metadata["trial_number"] == metadata["trial_number"])
        & (selected_metadata["rep_number"] == 1)
    ]

    if len(reference) != 1:
        raise ValueError(
            "Expected exactly one selected rep1 reference for "
            f"participant={metadata['participant']}, "
            f"block={metadata['block']}, "
            f"trial={metadata['trial_number']}, "
            f"but found {len(reference)}."
        )

    reference_row = reference.iloc[0]

    corrected["target_object_corrected"] = int(reference_row["target_object_final"])

    corrected["target_correction_applied"] = True
    corrected["target_reference_rep"] = 1
    corrected["target_reference_file"] = reference_row["file"]

    return corrected


# ------------------------------------------------------------
# Deriving Experimental Mapping (metadata)
# ------------------------------------------------------------


def derive_experimental_mapping(metadata):
    """
    Derive the classification label and board sensor from
    corrected recording metadata.

    B0:
        class label = trial_number
        board sensor = 3

    B1-B4:
        class label = corrected target_object
        board sensor = trial_number
    """

    mapped = metadata.copy()

    block = int(metadata["block"])

    trial_number = int(metadata["trial_number"])

    if not 0 <= trial_number <= 7:
        raise ValueError(f"Unexpected trial_number: {trial_number}")

    if block == 0:
        class_label = trial_number
        board_sensor_index = 3

        class_label_source = "trial_number"
        board_sensor_source = "fixed_B0_sensor_3"

    elif block in [1, 2, 3, 4]:
        if "target_object_corrected" not in metadata:
            raise ValueError("Corrected target metadata is required for blocks B1-B4.")

        class_label = int(metadata["target_object_corrected"])

        if not 0 <= class_label <= 7:
            raise ValueError(f"Unexpected corrected target: {class_label}")

        board_sensor_index = trial_number

        class_label_source = "target_object_corrected"

        board_sensor_source = "trial_number"

    else:
        raise ValueError(f"Unexpected block: {block}")

    mapped["class_label"] = class_label

    mapped["class_label_source"] = class_label_source

    mapped["board_sensor_index"] = board_sensor_index

    mapped["board_column"] = f"board_data_{board_sensor_index}"

    mapped["board_sensor_source"] = board_sensor_source

    return mapped


# ---------------------------------------------------------------------
# Trajectory resampling
# ---------------------------------------------------------------------


def resample_trajectory(
    values,
    n_timepoints=N_TIMEPOINTS,
):
    """
    Resample a trajectory to a fixed number of timestamps.

    Parameters
    ----------
    values : np.ndarray
        Shape (T, C)
        T = original number of timestamps
        C = number of channels

    n_timepoints : int
        Desired number of timestamps.

    Returns
    -------
    np.ndarray
        Shape (C, n_timepoints)
    """

    values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array (time, channels), got {values.shape}")

    if len(values) < 2:
        raise ValueError("Trajectory must contain at least two timestamps.")

    old_time = np.linspace(
        0.0,
        1.0,
        num=len(values),
    )

    new_time = np.linspace(
        0.0,
        1.0,
        num=n_timepoints,
    )

    resampled = np.stack(
        [
            np.interp(
                new_time,
                old_time,
                values[:, channel],
            )
            for channel in range(values.shape[1])
        ],
        axis=0,
    )

    return resampled.astype(np.float32)


# ---------------------------------------------------------------------
# Original preprocessing helpers
# ---------------------------------------------------------------------


def first_change_index(time_series, threshold):
    """
    Return the index immediately after the first sample whose
    absolute value exceeds the threshold.

    Reproduces the helper used in the original RTG preprocessing.
    """

    changes = np.where(np.abs(time_series) > threshold)[0]

    if changes.size == 0:
        return None

    return int(changes[0] + 1)


def transform_data_to_speed_profile(data, timestamps):
    """
    Compute the movement speed profile from positional data.

    Reproduces the helper used in the original RTG preprocessing.
    """

    gradient = np.gradient(
        data,
        timestamps,
        axis=0,
    )

    if data.ndim == 1:
        speed = np.abs(gradient)
    else:
        speed = np.linalg.norm(
            gradient,
            axis=1,
        )

    return speed


def replace_with_highest(array, target):
    """
    Replace occurrences of a target value with the largest
    non-target value in the array.

    Reproduces the helper used in the original RTG preprocessing.
    """

    if array is None or array.size == 0:
        return array

    mask = array != target

    if not np.any(mask):
        return array

    max_non_target = np.max(array[mask])

    return np.where(
        array == target,
        max_non_target,
        array,
    )


# ---------------------------------------------------------------------
# B0 movement-boundary logic
# ---------------------------------------------------------------------


def find_b0_movement_bounds(df):
    """
    Reproduce the B0 movement-boundary logic from
    Jonas' extract_Xy_movement.py.

    B0 uses:
        - all 75 glove coordinates for the speed profile
        - board_data_3 for object movement
        - 20% of maximum relative board displacement for `end`
        - 10% of maximum pre-end speed for `start`
        - 50 additional samples after `end` for movement export
    """

    glove_time = df["glove_timestamps"].to_numpy(dtype=float)

    glove_data = df[ALL_GLOVE_COLUMNS].to_numpy(dtype=float)

    board_columns = [f"board_data_{i}" for i in range(8)]

    board_data = df[board_columns].to_numpy(dtype=float)

    # Match the original preprocessing of invalid ToF values
    board_data = replace_with_highest(
        board_data,
        target=8191.0,
    )

    board_data = replace_with_highest(
        board_data,
        target=8190.0,
    )

    # ---------------------------------------------------------
    # Object movement endpoint
    # ---------------------------------------------------------

    board_data_mod = board_data[:, 3] - board_data[0, 3]

    end_threshold = 0.2 * board_data_mod.max()

    end = first_change_index(
        board_data_mod,
        end_threshold,
    )

    if end is None:
        raise ValueError("Could not detect B0 movement end.")

    # ---------------------------------------------------------
    # Hand movement start
    # ---------------------------------------------------------

    speed = transform_data_to_speed_profile(
        glove_data,
        glove_time,
    )

    if end <= 0:
        raise ValueError(f"Invalid B0 movement end: {end}")

    start_threshold = 0.1 * speed[:end].max()

    start = first_change_index(
        speed,
        start_threshold,
    )

    if start is None:
        raise ValueError("Could not detect B0 movement start.")

    if start >= end:
        raise ValueError(f"B0 start >= end: start={start}, end={end}")

    # Jonas' movement-export script retains 50 samples
    # beyond the detected endpoint.
    export_end = end + 50

    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "export_end_idx": int(export_end),
        "start_threshold": float(start_threshold),
        "end_threshold": float(end_threshold),
        "speed_max_before_end": float(speed[:end].max()),
        "board_max_displacement": float(board_data_mod.max()),
    }


# ---------------------------------------------------------------------
# B0 movement trajectory extraction
# ---------------------------------------------------------------------


def extract_b0_trajectory(df):
    """
    Extract the B0 movement trajectory using the original
    B0 movement-export definition.

    B0:
        start = glove-speed threshold
        detected end = 20% board_data_3 threshold
        export end = detected end + 50 samples

    Only the five fingertip sensors are retained,
    resulting in 15 channels.
    """

    bounds = find_b0_movement_bounds(df)

    start = bounds["start_idx"]
    export_end = bounds["export_end_idx"]

    fingertip_data = df[FINGERTIP_COLUMNS].to_numpy(dtype=np.float32)

    trajectory = fingertip_data[start:export_end]

    if trajectory.ndim != 2:
        raise ValueError(f"Expected 2D B0 trajectory, got {trajectory.shape}")

    if trajectory.shape[1] != 15:
        raise ValueError(f"Expected 15 fingertip channels, got {trajectory.shape[1]}")

    if len(trajectory) < 2:
        raise ValueError(f"B0 trajectory is too short: {len(trajectory)} samples.")

    if not np.isfinite(trajectory).all():
        raise ValueError("B0 trajectory contains non-finite values.")

    return trajectory, bounds


# ---------------------------------------------------------------------
# B1-B4 movement-boundary logic
# ---------------------------------------------------------------------


def find_b1_b4_movement_bounds(
    df,
    board_sensor_index,
):
    """
    Reproduce the B1-B4 reach-to-grasp movement boundaries
    used in Jonas' Fig17.py.

    B1-B4 use:
        - first hand_active == 1 for movement start
        - board_data_{trial_number} for object movement
        - 10% of maximum relative board displacement for object lift
        - no additional post-lift samples
    """

    board_sensor_index = int(board_sensor_index)

    if not 0 <= board_sensor_index <= 7:
        raise ValueError(f"Unexpected board sensor index: {board_sensor_index}")

    # ---------------------------------------------------------
    # Hand movement start
    # ---------------------------------------------------------

    hand_active = df["hand_active"].to_numpy()

    active_indices = np.where(hand_active == 1)[0]

    if len(active_indices) == 0:
        raise ValueError(
            "Could not detect B1-B4 movement start: no hand_active == 1 sample."
        )

    start = int(active_indices[0])

    # ---------------------------------------------------------
    # Object movement endpoint
    # ---------------------------------------------------------

    board_columns = [f"board_data_{i}" for i in range(8)]

    board_data = df[board_columns].to_numpy(dtype=float)

    # Reproduce Fig17.py
    board_data = replace_with_highest(
        board_data,
        target=8190.0,
    )

    # Fig17.py considers the target board signal only
    # after hand movement has started and excludes the
    # final recording sample.
    board_signal = board_data[
        start:-1,
        board_sensor_index,
    ]

    if len(board_signal) < 2:
        raise ValueError("B1-B4 board signal is too short after movement start.")

    # Zero-baseline the board signal at movement start.
    board_signal = board_signal - board_signal[0]

    end_threshold = 0.1 * board_signal.max()

    t_lift = first_change_index(
        board_signal,
        end_threshold,
    )

    if t_lift is None:
        raise ValueError("Could not detect B1-B4 object lift.")

    # t_lift is relative to `start`.
    end = int(start + t_lift)

    if start >= end:
        raise ValueError(f"B1-B4 start >= end: start={start}, end={end}")

    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "t_lift_relative": int(t_lift),
        "end_threshold": float(end_threshold),
        "board_max_displacement": float(board_signal.max()),
        "board_sensor_index": board_sensor_index,
    }


# ---------------------------------------------------------------------
# B1-B4 movement trajectory extraction
# ---------------------------------------------------------------------


def extract_b1_b4_trajectory(
    df,
    board_sensor_index,
):
    """
    Extract the B1-B4 reach-to-grasp fingertip trajectory.

    Reproduces the movement interval used in Jonas' Fig17.py:

        start = first hand_active == 1
        end   = object lift from the 10% target-board threshold

    Only the five fingertip sensors are retained, giving
    15 channels total.

    Returns
    -------
    trajectory : np.ndarray
        Shape (T, 15), before fixed-length resampling.

    bounds : dict
        Movement-boundary information returned by
        find_b1_b4_movement_bounds().
    """

    bounds = find_b1_b4_movement_bounds(
        df=df,
        board_sensor_index=board_sensor_index,
    )

    start = bounds["start_idx"]
    end = bounds["end_idx"]

    fingertip_data = df[FINGERTIP_COLUMNS].to_numpy(dtype=np.float32)

    # Matches Fig17.py:
    #
    # glove_data[start:start+t_lift]
    #
    # Since end = start + t_lift, this is equivalent to:
    #
    # fingertip_data[start:end]
    trajectory = fingertip_data[start:end]

    if trajectory.ndim != 2:
        raise ValueError(f"Expected 2D B1-B4 trajectory, got {trajectory.shape}")

    if trajectory.shape[1] != 15:
        raise ValueError(f"Expected 15 fingertip channels, got {trajectory.shape[1]}")

    if len(trajectory) < 2:
        raise ValueError(f"B1-B4 trajectory is too short: {len(trajectory)} samples.")

    if not np.isfinite(trajectory).all():
        raise ValueError("B1-B4 trajectory contains non-finite values.")

    return trajectory, bounds


# ---------------------------------------------------------------------
# ⚝⚝⚝⚝⚝ Unified recording-level trajectory extraction ⚝⚝⚝⚝⚝
# ---------------------------------------------------------------------


def extract_recording_trajectory(
    df,
    metadata,
):
    """
    Extract one movement trajectory according to its block.

    B0 uses the B0 movement-export definition.

    B1-B4 use the reach-to-grasp definition from Fig17.py.
    """

    block = int(metadata["block"])

    if block == 0:
        trajectory, bounds = extract_b0_trajectory(df)

        extraction_method = "B0_speed_board20_plus50"

    elif block in [1, 2, 3, 4]:
        trajectory, bounds = extract_b1_b4_trajectory(
            df=df,
            board_sensor_index=metadata["board_sensor_index"],
        )

        extraction_method = "B1-B4_hand_active_board10"

    else:
        raise ValueError(f"Unexpected block: {block}")

    return (
        trajectory,
        bounds,
        extraction_method,
    )


# ---------------------------------------------------------------------
# Unified reach-to-grasp dataset builder
# ---------------------------------------------------------------------


def build_reach_to_grasp_dataset(
    data_root=DATA_ROOT,
    n_timepoints=N_TIMEPOINTS,
):
    """
    Build the complete CTM-ready reach-to-grasp dataset.

    Returns
    -------
    X : np.ndarray
        Shape (N, 15, n_timepoints)

    y : np.ndarray
        Shape (N,)

    metadata_df : pandas.DataFrame
        One row per retained recording.

    excluded_df : pandas.DataFrame
        Logical trial slots without a successful recording.
    """

    # ---------------------------------------------------------
    # 1. Inventory
    # ---------------------------------------------------------

    inventory_df = build_recording_inventory(data_root)

    # ---------------------------------------------------------
    # 2. Raw metadata
    # ---------------------------------------------------------

    raw_metadata_df = pd.DataFrame(
        [extract_raw_metadata(path) for path in inventory_df["path"]]
    )

    # ---------------------------------------------------------
    # 3. Successful-attempt selection
    # ---------------------------------------------------------

    selected_df, excluded_df = select_successful_recordings(raw_metadata_df)

    # ---------------------------------------------------------
    # 4. B1-B4 repetition-0 correction
    # ---------------------------------------------------------

    corrected_df = pd.DataFrame(
        [
            apply_rep0_target_correction(
                row.to_dict(),
                selected_df,
            )
            for _, row in selected_df.iterrows()
        ]
    )

    # ---------------------------------------------------------
    # 5. Class and board mapping
    # ---------------------------------------------------------

    mapped_df = pd.DataFrame(
        [
            derive_experimental_mapping(row.to_dict())
            for _, row in corrected_df.iterrows()
        ]
    )

    # ---------------------------------------------------------
    # 6. Movement extraction + resampling
    # ---------------------------------------------------------

    X = []
    y = []
    output_metadata = []

    for _, row in mapped_df.iterrows():
        df = pd.read_csv(row["path"])

        (
            trajectory,
            bounds,
            extraction_method,
        ) = extract_recording_trajectory(
            df=df,
            metadata=row,
        )

        resampled = resample_trajectory(
            trajectory,
            n_timepoints=n_timepoints,
        )

        if resampled.shape != (
            15,
            n_timepoints,
        ):
            raise ValueError(
                f"Unexpected resampled shape for {row['file']}: {resampled.shape}"
            )

        if not np.isfinite(resampled).all():
            raise ValueError(f"Non-finite resampled trajectory for {row['file']}")

        X.append(resampled)

        y.append(int(row["class_label"]))

        record = row.to_dict()

        record["movement_start_idx"] = bounds["start_idx"]

        record["movement_end_idx"] = bounds["end_idx"]

        # B0 exports beyond the detected endpoint.
        # B1-B4 ends at the detected lift point.
        if "export_end_idx" in bounds:
            effective_end = bounds["export_end_idx"]

        else:
            effective_end = bounds["end_idx"]

        record["trajectory_end_idx"] = effective_end

        record["detected_movement_length"] = bounds["end_idx"] - bounds["start_idx"]

        record["trajectory_length"] = len(trajectory)

        record["extraction_method"] = extraction_method

        output_metadata.append(record)

    # ---------------------------------------------------------
    # 7. Final arrays
    # ---------------------------------------------------------

    X = np.stack(
        X,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    output_metadata_df = pd.DataFrame(output_metadata)

    return (
        X,
        y,
        output_metadata_df,
        excluded_df,
    )

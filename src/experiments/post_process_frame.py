# -*- coding: utf-8 -*-
"""

@author: jonas

Script for loading, cleaning, and preprocessing raw participant glove and board
data. It standardizes column names, applies rotation normalization, handles
missing values, corrects timestamps, and saves processed CSVs for later model
training and evaluation.

Relies on utilities defined in `utils.py`.
"""

import pandas as pd
import numpy as np
import utils

# -------------------------------------------------------------------------
# Configuration and constants
# -------------------------------------------------------------------------

num_sensors = 8
num_gestures = 8
labels = utils.object_list  # gesture/object labels from utility module

# Color definitions for plots or labeling consistency
colors = {
    0: 'blue',           # bold and deep
    1: 'orange',         # warm and vibrant
    2: 'green',          # calm and nature-inspired
    3: 'gray',           # neutral and balanced
    4: 'red',            # strong and attention-grabbing
    5: 'purple',         # soft and distinctive
    6: 'cyan',           # bright and clear
    7: 'black',          # standard neutral
    8: 'teal',           # modern and refreshing
    9: 'pink'            # playful and elegant
}

# Sensor permutation sets for handling mirrored or rotated trial configurations
perm0 = [0, 1, 2, 3, 4, 5, 6, 7]
perm1 = [0, 1, 2, 3, 4, 5, 6, 7]
perm2 = perm1[::-1]
perm3 = [3, 2, 1, 0, 7, 6, 5, 4]
perm4 = perm3[::-1]
perms = [perm0, perm1, perm2, perm3, perm4]

# -------------------------------------------------------------------------
# Column definitions
# -------------------------------------------------------------------------

# Raw and standardized column names
raw_glove_data_columns = [f'glove_data_{i}' for i in range(75)]
board_data_columns = [f'board_data_{i}' for i in range(num_sensors)]
sensor_data_columns = [
    f'SensorID_{i}_{axis}' for i in range(25) for axis in ['x', 'y', 'z']
]

# -------------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------------

participant_raw_data_path = "../../data/dataframes/"
participant_save_data_path = "../../data/dataframes/"

# -------------------------------------------------------------------------
# Main processing loop
# -------------------------------------------------------------------------

# Loop through all participant datasets
for name_counter, name in enumerate(
    ['participant_1', 'participant_2', 'participant_3', 'participant_4', 'participant_5']
):
    
    print(name)
    filename = name + '.csv'
    df = utils.load_dataframe_from_csv(participant_raw_data_path + filename)
    # ---------------------------------------------------------------------
    # Standardize column naming across possible dataset versions
    # ---------------------------------------------------------------------
    # try:
    #     df = df.rename(columns={'Block': 'block'})
    #     rename_map = dict(zip(raw_glove_data_columns, sensor_data_columns))
    #     df = df.rename(columns=rename_map)
    # except Exception:
    #     continue

    # try:
    #     df = df.rename(columns={'phase': 'block'})
    #     rename_map = dict(zip(raw_glove_data_columns, sensor_data_columns))
    #     df = df.rename(columns=rename_map)
    # except Exception:
    #     continue

    # # Drop obsolete column if present
    # try:
    #     df = df.drop('lifted_object', axis=1)
    # except Exception:
    #     continue

    # Final renaming step for safety
    # rename_map = {
    #     old: new for old, new in zip(raw_glove_data_columns, sensor_data_columns)
    #     if old in df.columns
    # }
    # df = df.rename(columns=rename_map)

    # Save standardized dataset
    # utils.save_dataframe_to_csv(df, participant_save_data_path + name + ".csv")

    # Reload saved file (ensures consistency of format)
    # df = utils.load_dataframe_from_csv(participant_save_data_path + name + ".csv")

    # Initialize empty DataFrame for processed data
    df_postprocessed = utils.create_pandas_frame()

    # Extract glove sensor data subset
    glove_data_subset = df[sensor_data_columns].to_numpy()
    df[sensor_data_columns] = glove_data_subset

    print(df.shape)

    # ---------------------------------------------------------------------
    # Iterate through experiment phases, trials, repetitions, and attempts
    # ---------------------------------------------------------------------
    for phase in range(df['block'].max() + 1):
        print(f"Processing phase {phase}")
        for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
            for rep_counter in range(df[
                (df['block'] == phase) &
                (df['trial_number'] == trial_number)
            ]['rep_number'].max() + 1):
                for attempt_counter in range(df[
                    (df['block'] == phase) &
                    (df['trial_number'] == trial_number) &
                    (df['rep_number'] == rep_counter)
                ]['attempt_number'].max() + 1):

                    # Select subset of dataframe for the given trial instance
                    df_att = df[
                        (df['block'] == phase) &
                        (df['trial_number'] == trial_number) &
                        (df['rep_number'] == rep_counter) &
                        (df['attempt_number'] == attempt_counter)
                    ]

                    mask = (
                        (df['block'] == phase) &
                        (df['trial_number'] == trial_number) &
                        (df['rep_number'] == rep_counter) &
                        (df['attempt_number'] == attempt_counter)
                    )

                    # -----------------------------------------------------------------
                    # Fill NaN trial success values
                    # -----------------------------------------------------------------
                    if np.isnan(df_att['trial_success'].to_numpy()[-1]):
                        df_att.loc[mask, 'trial_success'] = df_att.loc[mask, 'trial_success'].fillna(0)
                    else:
                        df_att.loc[mask, 'trial_success'] = df_att.loc[mask, 'trial_success'].fillna(1)

                    # -----------------------------------------------------------------
                    # Apply rotation normalization to glove data
                    # -----------------------------------------------------------------
                    glove_data_subset = df_att[sensor_data_columns].to_numpy()
                    print(glove_data_subset.shape)
                    glove_data_subset = utils.preprocess_data_rotation(glove_data_subset)

                    # Normalize timestamps to start from zero
                    glove_time = df_att['glove_timestamps'].to_numpy()
                    df_att.loc[mask, 'glove_timestamps'] = glove_time[:] - glove_time[0]

                    board_time = df_att['board_timestamps'].to_numpy()
                    df_att.loc[mask, 'board_timestamps'] = board_time[:] - board_time[0]

                    palm_time = df_att['palm_timestamps'].to_numpy()
                    df_att.loc[mask, 'palm_timestamps'] = palm_time[:] - palm_time[0]

                    # Update normalized data and remapped trial target
                    df_att.loc[mask, sensor_data_columns] = glove_data_subset
                    df_att.loc[mask, 'target_object'] = df.loc[
                        mask, 'target_object'
                    ].replace({trial_number: perms[int(phase)][int(trial_number)]})

                    # -----------------------------------------------------------------
                    # Detect and handle time gaps (e.g., dropped samples)
                    # -----------------------------------------------------------------
                    gaps = utils.find_large_time_gaps(glove_time, 300)

                    # -----------------------------------------------------------------
                    # Apply manual corrections and trimming for known problematic data
                    # -----------------------------------------------------------------
                    if name_counter == 0 and phase == 0 and trial_number == 7 and rep_counter == 5:
                        df_att = df_att.iloc[0:800]
                    if name_counter == 0 and phase == 3 and trial_number == 4 and rep_counter == 2:
                        df_att = df_att.iloc[0:800]
                    if name_counter == 0 and phase == 3 and trial_number == 4 and rep_counter == 3:
                        df_att = df_att.iloc[0:1000]
                    if name_counter == 0 and phase == 4 and trial_number == 3 and rep_counter == 1:
                        df_att = df_att.iloc[0:800]
                    if name_counter == 2 and phase == 0 and trial_number == 3 and rep_counter == 1:
                        df_att['trial_success'] = 0
                    if name_counter == 1 and phase == 0 and trial_number == 4 and rep_counter == 0:
                        df_att['trial_success'] = 0

                    # Append to postprocessed DataFrame
                    if gaps.size == 0:
                        df_postprocessed = pd.concat([df_postprocessed, df_att])
                    else:
                        index = gaps[0]
                        df_after = df_att.iloc[index + index + 1:]
                        df_postprocessed = pd.concat([df_postprocessed, df_after])

    # ---------------------------------------------------------------------
    # Final cleanup and save processed participant file
    # ---------------------------------------------------------------------
    df_postprocessed['participant'] = name_counter + 1

    # Rename columns to standardized glove data names
    rename_map = dict(zip(sensor_data_columns, utils.glove_data_columns))
    df_postprocessed = df_postprocessed.rename(columns=rename_map)
    utils.save_dataframe_to_csv(
        df_postprocessed,
        participant_save_data_path + "processed_" + name + ".csv"
    )

    print('Saved processed data for', name)
    # del df

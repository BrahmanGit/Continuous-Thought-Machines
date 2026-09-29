# -*- coding: utf-8 -*-
"""

@author: jonas

Script for extracting grasp and rest samples from preprocessed participant
datasets. It computes movement durations, identifies start/end indices for
grasp actions based on motion speed and board displacement, and saves
classified samples for downstream modeling (e.g., gesture recognition).

Relies on utility functions from `utils.py`.
"""

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import utils
# %matplotlib inline

# -------------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------------
participant_data_path = "../../data/dataframes/"
grasp_data_path = "../results/grasping_data/"

# -------------------------------------------------------------------------
# Main processing loop
# -------------------------------------------------------------------------

# Iterate through all participants
for participant in [
    'participant_1', 'participant_2', 'participant_3', 'participant_4', 'participant_5'
]:
    name = participant
    # Load preprocessed participant data
    df = utils.load_dataframe_from_csv(
        participant_data_path + 'processed_' + name + ".csv"
    )

    movement_time_list = []  # Stores movement durations for summary
    data_buffer = []         # Accumulates grasp/rest samples for export

    for phase in range(1):
        # Iterate through all trials for the given phase
        for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
            trial_movement_length = []
            for rep_counter in range(df[
                (df['block'] == phase) &
                (df['trial_number'] == trial_number)
            ]['rep_number'].max() + 1):

                # Extract one trial repetition subset
                df_att = df[
                    (df['block'] == phase) &
                    (df['trial_number'] == trial_number) &
                    (df['rep_number'] == rep_counter)
                ]

                # Process only successful trials
                if df_att['trial_success'].to_numpy()[-1] == 1:

                    # -----------------------------------------------------------------
                    # Extract relevant columns
                    # -----------------------------------------------------------------
                    glove_time = df_att['glove_timestamps'].to_numpy()
                    glove_data = df_att[utils.glove_data_columns].to_numpy()
                    board_data = df_att[utils.board_data_columns].to_numpy()
                    board_data = utils.replace_with_highest(arr=board_data, target=8191.0)  # Replace sensor saturation value
                    board_data = utils.replace_with_highest(arr=board_data, target=8190.0)  # Replace near-saturation value
                    board_time = df_att['board_timestamps'].to_numpy()
                    palm_data = df_att['palm_data'].to_numpy()
                    palm_time = df_att['palm_timestamps'].to_numpy()

                    # -----------------------------------------------------------------
                    # Compute motion speed from glove sensor trajectories
                    # -----------------------------------------------------------------
                    speed = utils.transform_data_to_speed_profile(glove_data, glove_time)

                    # -----------------------------------------------------------------
                    # Detect start and end of movement
                    # -----------------------------------------------------------------
                    board_data_mod = board_data[:, 3] - board_data[0, 3]
                    end = utils.first_change_index(
                        board_data_mod, 0.2 * board_data_mod.max()
                    )
                    start = utils.first_change_index(
                        speed, 0.1 * speed[:end].max()
                    )
                    movement_time_list.append(abs(end - start))
                    trial_movement_length.append(abs(end - start))

                    # -----------------------------------------------------------------
                    # Extract grasp-phase samples
                    # -----------------------------------------------------------------
                    
                    for elem in glove_data[start:int(end+50)]:
                        new_entry = list(elem) + [trial_number] + [rep_counter]
                        data_buffer.append(new_entry)

                    # -----------------------------------------------------------------
                    # Extract rest-phase samples (before movement start)
                    # -----------------------------------------------------------------
                    palm_data = -palm_data + palm_data.max()  # invert for alignment
                    indices = np.arange(0, start)

                    for elem in glove_data[max(0, start-100):start]:
                        new_entry = list(elem) + [8] + [rep_counter]  # class 8 = "rest"
                        data_buffer.append(new_entry)
    # ---------------------------------------------------------------------
    # Summary and save results
    # ---------------------------------------------------------------------
    print(f"Avg movement duration {name}: ", np.mean(movement_time_list))

    # Create dataframe of all collected samples
    grasp_df = pd.DataFrame(
        data_buffer,
        columns=(utils.glove_data_columns + ['class', 'rep'])
    )

    # Save grasp dataset for this participant
    utils.save_dataframe_to_csv(grasp_df, grasp_data_path + name + '_Xy_movement.csv')

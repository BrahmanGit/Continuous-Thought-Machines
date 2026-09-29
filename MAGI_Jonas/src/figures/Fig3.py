import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
from functools import partial
import models_jit
import models
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Initialize empty list to store parameter dictionaries for parallel processing
params_list = []

# Define data paths for input and output files
grasp_data_path = "../results/grasping_data/"
participant_data_path = "../../data/dataframes/"
results_data_path = "../results/predictions/"

target_names = utils.object_list[:]
# Time step size in milliseconds (or relevant time unit)
dt = 0.015
# Length of the integration window to update beliefs of MAGI

# Mean movement durations (in same time units as dt) for each of the 5 participants
mov_dur_list = [68.68, 81.07, 78.44, 100.4, 91.07]  # results obtained from measure_reach_duration_extract_Xy.py
finger_config = [0, 1]
# Multipliers for calculating beta (gravitational constant) parameter
beta_multipliers = np.array([0.15])

# Loop through all 5 participants
for name_c, name in enumerate(['participant_2']):

    # Load grasp data from CSV file for current participant
    grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')

    # Extract feature matrix X (glove sensor data) from dataframe
    X = grasp_df[utils.glove_data_columns].to_numpy()

    # Extract target labels y (grasp class labels) from dataframe
    y = grasp_df['class'].to_numpy()

    # Get mean movement duration for current participant
    mean_dur = mov_dur_list[name_c]

    lists, strings = utils.generate_finger_subsets()

    # Loop through beta multipliers (currently only one: 0.15)
    for counter, multiplier in enumerate(beta_multipliers):
            # Preprocess data by selecting only specified finger subset
            Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=finger_config)
            beta = 1 / (mean_dur * multiplier * dt)
            df_filename = participant_data_path + 'processed_' + name + '.csv'  # Input data file path
            df = utils.load_dataframe_from_csv(df_filename)
            for phase in range(1, df['block'].max() + 1):
                for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
                    for rep_counter in range(df[(df['block'] == phase) &
                                                (df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                                    if phase == 3 and trial_number == 3 and rep_counter == 3:
                                        df_att = df[(df['block'] == phase) &
                                                    (df['trial_number'] == trial_number) &
                                                    (df['rep_number'] == rep_counter)]

                                        # Extract glove sensor data and timestamps
                                        glove_time = df_att['glove_timestamps'].to_numpy()
                                        glove_data = df_att[utils.glove_data_columns].to_numpy()

                                        # Extract force sensor board data and clean outliers
                                        board_data = df_att[utils.board_data_columns].to_numpy()
                                        board_data = utils.replace_with_highest(arr=board_data, target=8191.0)
                                        board_data = utils.replace_with_highest(arr=board_data, target=8190.0)
                                        board_time = df_att['board_timestamps'].to_numpy()

                                        # Extract palm sensor data, clean outliers, and normalize to start at zero
                                        palm_data = df_att['palm_data'].to_numpy()
                                        palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
                                        palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
                                        palm_data = palm_data - palm_data[0]
                                        palm_time = df_att['palm_timestamps'].to_numpy()

                                        # Get true target object that was grasped
                                        y_true = int(df_att['target_object'].to_numpy()[-1])

                                        # Set analysis window start point
                                        start = 0

                                        # Get lifted target object
                                        lifted_target = int(df_att['target_object'].to_numpy()[-1])

                                        end = -1  # Use all data up to last point

                                        # Extract board sensor time series for current trial and normalize
                                        board_ts = board_data[start:end, int(trial_number)]
                                        board_ts = board_ts - board_ts[0]

                                        # Find object lift-off time (10% of max force threshold)
                                        t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max() * 0.1)

                                        # Find object put-down time by analyzing time series in reverse
                                        board_ts_flip = board_ts[::-1] - board_ts[end]
                                        t_down = len(board_ts) - utils.first_change_index(time_series=board_ts_flip, threshold=board_ts.max() * 0.1)
                                        if t_down <= t_lift: t_down = len(board_ts) - 1

                                        glove_data = glove_data[t_lift - 200:t_lift + 200]
                                        glove_time = glove_time[t_lift - 200:t_lift + 200]
                                        glove_data = utils.preprocess_data_finger_config(glove_data_subset=glove_data, finger_config=finger_config)
                                        print('Phase', phase, 'Trial', trial_number, 'Rep', rep_counter)

                                        integration_window_length = 5
                                        ncf = models.NearestCentroidClassifier()
                                        ncf.integration_window = integration_window_length
                                        ncf.fit(Xf, y)
                                        cens = []

                                        plotted_dim = np.arange(int(len(finger_config) * 3))
                                        glove_data = glove_data[:, plotted_dim]

                                        x0 = glove_data[0]
                                        Xff = Xf[:, plotted_dim]
                                        noise_KF, noise_obs = 10e-6, 10e-6
                                        grav_model = models_jit.create_dirichlet_model(
                                            X=Xff,
                                            y=y,
                                            noise_process=noise_KF,
                                            noise_obs=noise_obs,
                                            x0=x0,
                                            dt=dt,
                                            integration_window_length=integration_window_length,
                                            gravitational_const=0.125,
                                            reset_interval=20
                                        )

                                        nc = models.NearestCentroidClassifier()
                                        nc.integration_window = integration_window_length
                                        nc.fit(Xff, y)

                                        obs = glove_data

                                        # Define reference dimensions
                                        REFERENCE_WIDTH = 8  # inches
                                        REFERENCE_FONTSIZE = 15

                                        # For each script, calculate scaled font size
                                        fig_width = 8  # your actual figure width
                                        FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)

                                        fig, axs = plt.subplots(2, sharex=True, figsize=(fig_width, 8),
                                                                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

                                        grav_model = models_jit.create_dirichlet_model(
                                            X=Xff,
                                            y=y,
                                            noise_process=noise_KF,
                                            noise_obs=noise_obs,
                                            x0=obs[0],
                                            dt=dt,
                                            integration_window_length=integration_window_length,
                                            gravitational_const=beta,
                                            reset_interval=20
                                        )

                                        print('created')
                                        predicted_states1 = []
                                        means = []
                                        varis = []
                                        for observation in obs:
                                            mean, var, ev = grav_model.predict_step(observation)
                                            means.append(mean)
                                            varis.append(var)
                                            state = []
                                            for class_ in np.arange(9): state.append(grav_model.base_model_lst[class_].x)
                                            predicted_states1.append(state)

                                        means, varis = np.array(means), np.array(varis)

                                        class1 = y_true
                                        class2 = np.argsort(np.sum(means[:, 0:8], axis=0), axis=0)[-2]
                                        class3 = np.argsort(np.sum(means[:, 0:8], axis=0), axis=0)[-3]

                                        predicted_states1 = np.array(predicted_states1)

                                        # Contrasting line styles for class1 vs class2
                                        line_styles = ["-", "--"]  # solid for class1, dashed for class2
                                        line_widths = [2.5, 2.0]   # slightly thicker for class1

                                        # ---- Top plot: traces ----
                                        for dim in np.arange(int(len(finger_config) * 3)):
                                            axs[0].plot(glove_time, obs[:, dim], linestyle="-", color='gray', linewidth=2, alpha=0.5, zorder=0)
                                            for class_num, class_ in enumerate([class1, class2]):
                                                
                                                if dim == 0:
                                                    if class_num == 0: axs[0].plot(glove_time, obs[:, dim], linestyle="-", color='gray', linewidth=3, alpha=0.5, zorder=0, label='observation')
                                                    axs[0].plot(glove_time, obs[:, dim], linestyle="-", color='gray', linewidth=3, alpha=0.5, zorder=0)
                                                    axs[0].plot(glove_time, predicted_states1[:, class_num, dim],
                                                                linestyle=line_styles[class_num],
                                                                linewidth=line_widths[class_num],
                                                                alpha=1.0, color=utils.colors[class_],
                                                                label=f"$m_{class_}$ state estimate")
                                                    axs[0].scatter(glove_time[-1], nc.centroids[class_][dim],
                                                                   facecolor='white', edgecolor=utils.colors[class_],
                                                                   marker="o", s=40, linewidth=2,
                                                                   label=f"Grasp {class_} endpoint")
                                                else:
                                                    axs[0].plot(glove_time, predicted_states1[:, class_num, dim],
                                                                linestyle=line_styles[class_num],
                                                                linewidth=line_widths[class_num],
                                                                alpha=1.0, color=utils.colors[class_])
                                                    axs[0].scatter(glove_time[-1], nc.centroids[class_][dim],
                                                                   facecolor='white', edgecolor=utils.colors[class_],
                                                                   marker="o", s=40, linewidth=2)

                                        # ============================================================
                                        # INSET CONFIGURATION — adjust all parameters here freely
                                        # ============================================================
                                        
                                        # --- Data zoom window (which part of the time series to zoom into) ---
                                        zoom_start_idx = 200     # index of first time point shown in inset
                                        zoom_end_idx   = 250    # index of last time point shown in inset
                                        
                                        # y-axis zoom in DATA units — set both to None for auto-fit
                                        zoom_ymin = 0.043      # e.g. 0.2  — bottom of the visible y range in the inset
                                        zoom_ymax = 0.057       # e.g. 0.8  — top of the visible y range in the inset
                                        
                                        # --- Inset box position and size on the main axes ---
                                        # All values are in axes-fraction coordinates (0.0 = left/bottom, 1.0 = right/top)
                                        inset_x      = 0.60   # left edge of the inset box
                                        inset_y      = 0.77   # bottom edge of the inset box
                                        inset_width  = 0.30   # width of the inset box
                                        inset_height = 0.2   # height of the inset box
                                        
                                        # --- Connector lines between main plot and inset ---
                                        # loc1 / loc2 select which corners of the zoom rectangle connect to the inset
                                        # Corner codes: 1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right
                                        inset_loc1 = 2
                                        inset_loc2 = 3
                                        
                                        # ============================================================
                                        # Build the inset — no need to edit below this line
                                        # ============================================================
                                        axins = axs[0].inset_axes([inset_x, inset_y, inset_width, inset_height])
                                        
                                        for dim in np.arange(int(len(finger_config) * 3)):
                                            axins.plot(glove_time, obs[:, dim], linestyle="-", color='gray', linewidth=10.5, alpha=0.5, zorder=0)
                                            for class_num, class_ in enumerate([class1, class2]):
                                                axins.plot(glove_time, predicted_states1[:, class_num, dim],
                                                           linestyle=line_styles[class_num],
                                                           linewidth=line_widths[class_num],
                                                           alpha=1.0, color=utils.colors[class_])
                                        
                                        # Apply x-axis zoom
                                        x1 = glove_time[zoom_start_idx]
                                        x2 = glove_time[zoom_end_idx]
                                        axins.set_xlim(x1, x2)
                                        
                                        # Apply y-axis zoom
                                        if zoom_ymin is not None and zoom_ymax is not None:
                                            # Fully manual: you specify exact data-unit limits
                                            axins.set_ylim(zoom_ymin, zoom_ymax)
                                        elif zoom_ymin is not None or zoom_ymax is not None:
                                            # Semi-manual: one bound fixed, other auto-fitted from visible data
                                            visible_obs   = obs[zoom_start_idx:zoom_end_idx + 1]
                                            visible_preds = predicted_states1[zoom_start_idx:zoom_end_idx + 1]
                                            all_vals = np.concatenate([visible_obs.ravel(), visible_preds[:, [class1, class2], :].ravel()])
                                            pad = (all_vals.max() - all_vals.min()) * 0.05
                                            auto_min = all_vals.min() - pad
                                            auto_max = all_vals.max() + pad
                                            axins.set_ylim(
                                                zoom_ymin if zoom_ymin is not None else auto_min,
                                                zoom_ymax if zoom_ymax is not None else auto_max
                                            )
                                        else:
                                            # Fully auto: fit to whatever data is visible in the x-window
                                            visible_obs   = obs[zoom_start_idx:zoom_end_idx + 1]
                                            visible_preds = predicted_states1[zoom_start_idx:zoom_end_idx + 1]
                                            all_vals = np.concatenate([visible_obs.ravel(), visible_preds[:, [class1, class2], :].ravel()])
                                            pad = (all_vals.max() - all_vals.min()) * 0.05
                                            axins.set_ylim(all_vals.min() - pad, all_vals.max() + pad)
                                        
                                        axins.tick_params(axis='both', labelsize=FS * 0.85)
                                        mark_inset(axs[0], axins, loc1=inset_loc1, loc2=inset_loc2, fc="none", ec="0.5", linestyle="--")

                                        # ---- Bottom plot: beliefs (unchanged) ----
                                        for class_ in np.arange(9):
                                            axs[1].plot(glove_time, means[:, class_], linestyle="-",
                                                        color=utils.colors[class_], linewidth=1.5,
                                                        label=target_names[class_])
                                            axs[1].fill_between(glove_time,
                                                                means[:, class_] - varis[:, class_],
                                                                means[:, class_] + varis[:, class_],
                                                                color=utils.colors[class_], alpha=0.2)

                                        axs[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=FS)
                                        axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 2.5), ncol=1, fontsize=FS)
                                        axs[0].set_ylabel("$o_t, \hat{x}_t$ [cm]", fontsize=FS)
                                        axs[1].set_ylabel("$p(m_i|o_{1:\\tau})$", fontsize=FS)
                                        axs[1].set_xlabel('Time [ms]', fontsize=FS)
                                        for tick in axs[1].get_xticklabels():
                                            tick.set_fontsize(FS)
                                        for tick in axs[1].get_yticklabels():
                                            tick.set_fontsize(FS)
                                        for tick in axs[0].get_yticklabels():
                                            tick.set_fontsize(FS)
                                        plt.savefig('Fig3_rev.pdf', bbox_inches="tight")
                                        # plt.show()
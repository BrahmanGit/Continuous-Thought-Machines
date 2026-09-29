import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample


colors = {
    0: 'blue',           # #0072B2 - bold and deep
    1: 'orange',         # #E69F00 - warm and vibrant
    2: 'green',          # #009E73 - calm and nature-inspired
    3: 'gray',           # #999999 - neutral and balanced
    4: 'red',            # #D55E00 - strong and attention-grabbing
    5: 'purple',         # #CC79A7 - soft and distinctive
    6: 'cyan',           # #56B4E9 - bright and clear
    7: 'black',         # #F0E442 - bright and energetic
    8: 'teal',           # #1B9E77 - modern and refreshing
    9: 'pink'            # #E78AC3 - playful and elegant
}


object_list = utils.object_list 

# Data collection for both aperture and distance travelled
aperture_dict = []
distance_dict = []

data_path = "../../data/dataframes/"
grasp_data_path = "../results/grasping_data/"

FS = 15
for name_c, name in enumerate(['participant_1', 'participant_2', 'participant_3', 'participant_4', 'participant_5']):
    
    df = utils.load_dataframe_from_csv(data_path+"/processed_"+name+'.csv')
    
    for block in range(1, df['block'].max()+1):
        for trial_number in np.arange(df[(df['block'] == block)]['trial_number'].max()+1):        
            for rep_counter in range(df[(df['block'] == block) & (df['trial_number'] == trial_number)]['rep_number'].max()+1):                                 
                df_att = df[(df['block'] == block) & (df['trial_number'] == trial_number) & (df['rep_number'] == rep_counter) & (df['trial_success'] == 1)]    
                
                glove_time = df_att['glove_timestamps'].to_numpy()
                glove_data = df_att[utils.glove_data_columns].to_numpy()
                board_data = df_att[utils.board_data_columns].to_numpy()
                board_data = utils.replace_with_highest(arr=board_data, target=8190.0)
                board_time = df_att['board_timestamps'].to_numpy()
                palm_data = df_att['palm_data'].to_numpy()
                palm_time = df_att['palm_timestamps'].to_numpy()
                y_true = int(df_att['target_object'].to_numpy()[-1])
                # Common processing
                hand_active_indices = np.where(df_att['hand_active'] == 1)[0]
                if len(hand_active_indices) == 0:
                    continue
                end = -1
                board_ts = board_data[:, int(trial_number)]     
                # APERTURE DATA PROCESSING
                start_aperture = hand_active_indices[0]
                board_ts_aperture = board_data[start_aperture:end, int(trial_number)]
                board_ts_aperture = board_ts_aperture - board_ts_aperture[0]
                t_lift_aperture = utils.first_change_index(time_series=board_ts_aperture, threshold=board_ts_aperture.max()*0.1)
                mov_data_aperture = utils.preprocess_data_finger_config(glove_data[start_aperture:start_aperture+t_lift_aperture], [0,1,2,3,4])
                mov_data_aperture = resample(x=mov_data_aperture, num=100)
                dist_aperture = utils.compute_thumb_index_distance(mov_data_aperture[:,0:6])*100
                
                ap_dict = {
                    'ap': dist_aperture,
                    'target': y_true,
                    'participant': name_c
                }
                aperture_dict.append(ap_dict)
                # DISTANCE TRAVELLED DATA PROCESSING
                start_distance = max(0, hand_active_indices[0] - 30)
                board_ts_distance = board_data[start_distance:end, int(trial_number)]
                board_ts_distance = board_ts_distance - board_ts_distance[0]
                t_lift_distance = utils.first_change_index(time_series=board_ts_distance, threshold=board_ts_distance.max()*0.1)
                mov_data_distance = utils.preprocess_data_finger_config(glove_data[start_distance:start_distance+t_lift_aperture], [0,1,2,3,4])
                mov_data_distance = resample(x=mov_data_distance, num=101)
                dist_distance = utils.compute_distance_travelled_in_percent(mov_data_distance, target=mov_data_distance[-1])
                
                dist_dict = {
                    'ap': dist_distance,
                    'target': y_true,
                    'participant': name_c
                }
                distance_dict.append(dist_dict)

# Convert to DataFrames
df_aperture = pd.DataFrame(aperture_dict)
df_distance = pd.DataFrame(distance_dict)

# Create X-axes
x_aperture = np.linspace(0, 1, 100)
x_distance = np.linspace(0, 1, 100)

# Sort targets and participants
targets = sorted(df_aperture['target'].unique())
participants = sorted(df_aperture['participant'].unique())

# Create figure with 2 rows and len(targets) columns
num_targets = len(targets)
width_ratios = [1] * num_targets

# fig, axes = plt.subplots(2, num_targets, figsize=(20, 8),
#                          gridspec_kw={'width_ratios': width_ratios},
#                          sharey='row', sharex=True)

axis_label_style = {"fontsize": FS, "color": "black"}  # axis labels font size & color
tick_label_fontsize = FS    
title_fs = FS                           # tick labels font size
legend_fontsize = FS      
# Create figure with 2 rows and len(targets) columns
num_targets = len(targets)
width_ratios = [1] * num_targets

fig, axes = plt.subplots(2, num_targets, figsize=(18, 6),
                         gridspec_kw={'width_ratios': width_ratios},
                         sharey='row', sharex=True)

# Ensure axes is 2D array even with single column
if num_targets == 1:
    axes = axes.reshape(2, 1)

# Plot aperture data (top row)
for i, target in enumerate(targets):
    ax = axes[0, i]
    df_target = df_aperture[df_aperture['target'] == target]
    
    for participant in participants:
        df_part = df_target[df_target['participant'] == participant]
        
        if df_part.empty:
            continue
        
        try:
            apertures = np.vstack(df_part['ap'].to_numpy())
        except ValueError:
            continue
        
        mean_aperture = np.mean(apertures, axis=0)
        std_aperture = np.std(apertures, axis=0)
        ax.plot(x_aperture, mean_aperture, label=f'P{participant+1}', linewidth=2)
        ax.fill_between(x_aperture, mean_aperture - std_aperture, mean_aperture + std_aperture, alpha=0.3)
    
    ax.set_title(object_list[target], fontsize=title_fs)
    ax.grid(True)

    # Apply tick label font size
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    if i == 0:
        ax.set_ylabel('Aperture [cm]', **axis_label_style)

# Plot distance travelled data (bottom row)
for i, target in enumerate(targets):
    ax = axes[1, i]
    df_target = df_distance[df_distance['target'] == target]
    
    for participant in participants:
        df_part = df_target[df_target['participant'] == participant]
        if participant == 4: 
            print(df_part)
        
        if df_part.empty:
            continue
        
        try:
            distances = np.vstack(df_part['ap'].to_numpy())
        except ValueError:
            continue
        
        mean_distance = np.mean(distances, axis=0)
        std_distance = np.std(distances, axis=0)
        
        ax.plot(x_distance, mean_distance, label='P'+str(participant+1), linewidth=2)
        ax.fill_between(x_distance, mean_distance - std_distance, mean_distance + std_distance, alpha=0.3)
    
    ax.set_xlabel('Completion [ratio]', **axis_label_style)
    ax.grid(True)

    # Apply tick label font size
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    if i == 0:
        ax.set_ylabel('Distance [ratio]', **axis_label_style)

# Add legend outside the plot (right side)
handles, labels = axes[0, -1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.05, 0.5),
           borderaxespad=0, fontsize=FS)

plt.tight_layout()
plt.savefig('Fig18_rev.pdf', bbox_inches='tight')
# plt.show()
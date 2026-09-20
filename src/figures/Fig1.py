
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import numpy as np
import matplotlib.pyplot as plt
import models 
# Initialize empty list to store parameter dictionaries for parallel processing
params_list = []

# Define data paths for input and output files
grasp_data_path = "../results/grasping_data/"
participant_data_path = "../../data/dataframes/"
results_data_path = "../results/predictions/"

# Time step size in milliseconds (or relevant time unit)
dt = 0.015
# Length of the integration window to update beliefs of MAGI

# Mean movement durations (in same time units as dt) for each of the 5 participants
mov_dur_list = [68.68, 81.07, 78.44, 100.4, 91.07]#results obtained from measure_reach_duration_extract_Xy.py
finger_config = [0, 1, 2, 3, 4]
# Multipliers for calculating beta (gravitational constant) parameter
beta_multipliers = np.array([0.15])

# Loop through all 5 participants
speeds = []
times = []
class_ = 6
for name_c, name in enumerate(['participant_2']):#'participant_1', 'participant_2', 'participant_3', 
    
    # Load grasp data from CSV file for current participant
    grasp_df = utils.load_dataframe_from_csv(grasp_data_path+name+'_Xy.csv')
    
    # Extract feature matrix X (glove sensor data) from dataframe
    X = grasp_df[utils.glove_data_columns].to_numpy()
    Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=finger_config)   
    y = grasp_df['class'].to_numpy()
    
    nc = models.NearestCentroidClassifier()
    nc.integration_window = 10
    nc.fit(Xf, y)
    
    # Get mean movement duration for current participant
    mean_dur = mov_dur_list[name_c]
  
    lists, strings = utils.generate_finger_subsets()
   
    # Loop through beta multipliers (currently only one: 0.15)
    for counter, multiplier in enumerate(beta_multipliers):    
            # Preprocess data by selecting only specified finger subset
            Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=finger_config)   
            beta = 1/(mean_dur*multiplier*dt)        
            df_filename = participant_data_path+'processed_'+name+'.csv'  # Input data file path
            df = utils.load_dataframe_from_csv(df_filename)
            for phase in range(1, df['block'].max() + 1):
                for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
                    for rep_counter in range(df[(df['block'] == phase) &
                                                (df['trial_number'] == trial_number)]['rep_number'].max() + 1):
                                    
                                        df_att = df[(df['block'] == phase) &
                                                    (df['trial_number'] == trial_number) &
                                                    (df['rep_number'] == rep_counter)]
        
                                        # Extract glove sensor data and timestamps
                                        glove_time = df_att['glove_timestamps'].to_numpy()
                                        glove_data = df_att[utils.glove_data_columns].to_numpy()
                                        glove_data = utils.preprocess_data_finger_config(glove_data_subset=glove_data, finger_config=finger_config)
                                        # Extract force sensor board data and clean outliers
                                        board_data = df_att[utils.board_data_columns].to_numpy()
                                        board_data = utils.replace_with_highest(arr=board_data, target=8191.0)  # Replace sensor saturation value
                                        board_data = utils.replace_with_highest(arr=board_data, target=8190.0)  # Replace near-saturation value
                                        board_time = df_att['board_timestamps'].to_numpy()
                                        
                                        # Extract palm sensor data, clean outliers, and normalize to start at zero
                                        palm_data = df_att['palm_data'].to_numpy()
                                        palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
                                        palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
                                        palm_data = palm_data-palm_data[0]  # Zero-baseline the palm data
                                        palm_time = df_att['palm_timestamps'].to_numpy()
                                        
                                        # Get true target object that was grasped
                                        y_true = int(df_att['target_object'].to_numpy()[-1])
                                        if y_true == class_ and phase == 1 :
                                            # Set analysis window start point
                                            start = 0
                                    
                                            # Get lifted target object (same as y_true, redundant)
                                            lifted_target = int(df_att['target_object'].to_numpy()[-1])
                                   
                                            end = -1  # Use all data up to last point
                                            
                                            # Extract board sensor time series for current trial and normalize
                                            board_ts = board_data[start:end, int(trial_number)]
                                            board_ts = board_ts - board_ts[0]
                                            
                                            # Find object lift-off time (10% of max force threshold)
                                            t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)
                                            
                                            # Find object put-down time by analyzing time series in reverse
                                            board_ts_flip = board_ts[::-1]-board_ts[end]
                                            t_down = len(board_ts)-utils.first_change_index(time_series=board_ts_flip, threshold=board_ts.max()*0.1)
                                            if t_down <= t_lift: t_down = len(board_ts)-1  # Ensure t_down is after t_lift
                                            
                                           
                                            glove_data = glove_data[t_lift-200:t_lift+200]*100
                                            glove_time = glove_time[t_lift-200:t_lift+200]
                                            
                                            speed = utils.transform_data_to_speed_profile(glove_data, glove_time)
                                            print('app')
                                            speeds.append(speed)
                                            times.append(glove_time-glove_time[0])
                                            

# Define reference dimensions
REFERENCE_WIDTH = 8  # inches
REFERENCE_FONTSIZE = 15

# For each script, calculate scaled font size
fig_width = 8  # your actual figure width
FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)

fig, ax = plt.subplots(1, figsize=(fig_width, 5))

# First subplot - mean and std of aligned speeds
# Align all speed profiles
aligned_data = []
for time, speed in zip(times, speeds):
    max_idx = np.argmax(speed)
    time_aligned = time - time[max_idx]
    aligned_data.append((time_aligned, speed))

# Find the shortest profile length
min_length = min(len(speed) for _, speed in aligned_data)
all_aligned_times = []
for time_aligned, _ in aligned_data:
    all_aligned_times.extend(time_aligned[:min_length])

common_times = np.linspace(min(all_aligned_times), max(all_aligned_times), min_length)

speeds_interpolated = []
for time_aligned, speed in aligned_data:
    speed_interp = np.interp(common_times, time_aligned[:min_length], speed[:min_length])
    speeds_interpolated.append(speed_interp)

speeds_array = np.array(speeds_interpolated)
mean_speed = np.mean(speeds_array, axis=0)
std_speed = np.std(speeds_array, axis=0)

# Shift time to start at 0
common_times_shifted = common_times - common_times[0]

# # Plot with error bars
ax.plot(common_times_shifted, mean_speed, color='gray', label='Observed average speed')
# ax.errorbar(common_times_shifted, mean_speed, yerr=std_speed, capsize=3, capthick=1, linewidth=2, label='Mean ± Std', color='gray')
ax.fill_between(common_times_shifted, mean_speed - std_speed, mean_speed + std_speed, alpha=0.3, color='gray')

# Parameters
x0 = nc.centroids[8]
lambda_ = nc.centroids[class_]
lim = len(mean_speed)
dt = 0.015
tau = 100 * dt
beta_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

for beta in beta_values:
    x_sim = [x0]
    dx_dt = []
    
    for i in range(lim):
        xt1 = (lambda_ - x_sim[-1]) / (beta * tau) * dt + x_sim[-1]
        x_sim.append(xt1)
    
    t = np.arange(0, (lim + 1) * dt, dt)
    
    x_sim = np.array(x_sim)
    speed_sim = utils.transform_data_to_speed_profile(x_sim, np.linspace(0, len(x_sim)*dt, len(x_sim))*10)
    offset = np.argmax(mean_speed)
    time_offset = common_times_shifted[offset]
    time_sim = np.linspace(0, len(x_sim)*dt, len(x_sim))*1000+time_offset#*1000
    ax.plot(time_sim, speed_sim, label=f'β = {beta}', linewidth=2, linestyle='--')

ax.set_xlabel('Time [ms]', fontsize=FS)
ax.set_ylabel('Speed [cm/ms]', fontsize=FS)
ax.set_xticks(np.arange(0, 4000, 500))
ax.set_xticklabels(np.arange(0, 4000, 500), fontsize=FS)
ax.set_xlim([1800, 4000])
ax.set_yticks(np.round(np.linspace(0, max(mean_speed), 5), 5))
ax.set_yticklabels(np.round(np.linspace(0, max(mean_speed), 5), 5), fontsize=FS)
plt.legend(fontsize=FS)
plt.savefig('Fig1_rev.pdf', bbox_inches="tight")
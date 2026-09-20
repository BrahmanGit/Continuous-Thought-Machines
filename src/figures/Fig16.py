# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:24:01 2025

@author: jonas
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


num_sensors = 8
num_gestures = 8
mean_data_columns = [f'mean_pred_{i}' for i in range(9)]
var_data_columns = [f'var_pred_{i}' for i in range(9)]
nc_pred_data_columns = [f'nc_pred_{i}' for i in range(9)]   
labels = utils.object_list

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

perms = utils.perms

legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=label) for i, label in enumerate(labels) if i in colors]
lists, strings = utils.generate_finger_subsets()

audio_stimulus_offset = 2500 #offset from recording beginnning 

data_path = "../../data/dataframes/"
for name_counter, name in enumerate(['participant_1', 'participant_2', 'participant_3', 'participant_4', 'participant_5']):
    all_times = []    
    for finger_combination_list, finger_combination in zip(lists[-1:], strings[-1:]):
        
        df = utils.load_dataframe_from_csv(data_path+"/processed_"+name+'.csv')
        
        for block in np.arange(1, df['block'].max()+1):
            for trial_number in np.arange(df[(df['block'] == block)]['trial_number'].max()+1):        
                for rep_counter in np.arange(df[(df['block'] == block) & (df['trial_number'] == trial_number)]['rep_number'].max()+1):                                 
                 
                    attempt_counter = df[(df['block'] == block) & (df['trial_number'] == trial_number) & (df['rep_number'] == rep_counter)]['attempt_number'].max()
                    
                    df_att = df[(df['block'] == block) & (df['trial_number'] == trial_number) & (df['rep_number'] == rep_counter)]         
                    df_att = df[(df['block'] == block) & (df['trial_number'] == trial_number) & (df['rep_number'] == rep_counter) & (df['trial_success'] == 1)]   
                    
                    glove_time = df_att['glove_timestamps'].to_numpy()
                    glove_data = df_att[utils.glove_data_columns].to_numpy()
                    board_data = df_att[utils.board_data_columns].to_numpy()
                    board_data = utils.replace_with_highest(arr=board_data, target=8191.0)
                    board_data = utils.replace_with_highest(arr=board_data, target=8190.0)
                    # print(board_data.max())
                    board_time = df_att['board_timestamps'].to_numpy()
                    palm_data = df_att['palm_data'].to_numpy()
                    palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
                    palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
                    palm_data = palm_data-palm_data[0]
                    palm_time = df_att['palm_timestamps'].to_numpy()
                    board_time = board_time-board_time[0]
                    glove_time = glove_time-glove_time[0]
                    palm_time = palm_time-palm_time[0]
                    # print(palm_data.max())
                    y_true = int(df_att['target_object'].to_numpy()[-1])
                    start = 0
                    # start = np.where(df_att['hand_active'] == 1)[0][0]
            
                    lifted_target = int(df_att['target_object'].to_numpy()[-1])
                    
                    speed = utils.transform_data_to_speed_profile(glove_data, glove_time)
                    
                    end = -1
                    board_ts = board_data[start:end, int(trial_number)]
                    board_ts = board_ts - board_ts[0]
                    board_ts = np.abs(board_ts)
                    # t_lift = ManusMetaLib.first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)
                    t_lift = utils.first_crossing_index(series=board_ts, threshold=3, steps=3)
                    
                    
                    
                    board_ts_flip = board_ts[::-1]-board_ts[end]
                    t_down = len(board_ts)-utils.first_change_index(time_series=board_ts_flip, threshold=board_ts.max()*0.1)
                    if t_down <= t_lift: t_down = len(board_ts)-1
                    hand_lift = utils.first_change_index(time_series=palm_data, threshold=palm_data.max()*0.1)
                

                    
                    if finger_combination in ['01234'] and block == 1 and trial_number == 1 and rep_counter == 2 and name_counter == 0:
                        print('plot')
                        FS = 15
                        fig, ax = plt.subplots(2, sharex=True, figsize=(18, 6))
                        plt.title('P'+str(name_counter+1)+' phase '+str(block)+' trial '+str(trial_number)+' rep '+str(rep_counter))
                        lw = 3
                        for j in range(8):
                            if j == y_true:
                                ax[1].plot(palm_time, board_data[:, j], color=colors[j], linewidth=lw, label='Sensor '+str(j)) 
                            else:
                                ax[1].plot(palm_time, board_data[:, j], color=colors[j], label='Sensor '+str(j))  
                                
                        glove_data = utils.preprocess_data_finger_config(
                            glove_data_subset=glove_data,
                            finger_config=finger_combination_list
                        )*10
                        ax[0].plot(glove_time[:], glove_data[:])                            
                        ax[1].plot(palm_time, palm_data, color=colors[8], label='Rest sensor')
                                
                        
                        window_length = 10
                        kernel = np.ones(window_length) / window_length
                        moving_avg = np.apply_along_axis(
                            lambda col: np.convolve(col, kernel, mode="same"),
                            axis=0,
                            arr=board_data
                        )
                        t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)#TO-REMOVE
                        lift_detected = t_lift#TO-REMOVE
                        # lift_detected, vals = np.where(moving_avg > 100)
                        # lift_detected = lift_detected[0]
                       
                        board_block_index = perms[int(block)].index(y_true)
                        true_board_data = board_data[:,board_block_index]
                        true_board_data = true_board_data#-true_board_data[0]
                        audio_cue = np.where(palm_time>audio_stimulus_offset)[0][0]#-100
                        maximum_object_lift = np.where(true_board_data>100)[0][0]
    
                        audio_cue = np.where(palm_time>audio_stimulus_offset)[0][0]#-50
                        maximum_object_lift = np.where(true_board_data>0.3*true_board_data.max())[0][0]
                        'refinement'
                        t_lift = utils.first_crossing_index(series=board_ts[hand_lift:], threshold=7, steps=4)+hand_lift
                        # ax[1].axvline(x=glove_time[t_lift], color=colors[y_true], linewidth=3) 
                        
                        resc = 'red'
                        ax[0].axvline(x=glove_time[lift_detected]+audio_stimulus_offset, color=resc, label='Success audio cue')
                        ax[1].axvline(x=glove_time[lift_detected]+audio_stimulus_offset, color=resc)
                        ax[1].axvline(x=glove_time[lift_detected], color='green')
                        ax[1].axvline(x=glove_time[hand_lift], color='teal', linestyle='--')
                        
                        # Axis labels
                        ax[1].set_xlabel('Time [ms]', fontsize=FS)
                        ax[0].set_ylabel('Coordinates [cm]', fontsize=FS)
                        ax[1].set_ylabel('Range [mm]', fontsize=FS)
                        
                        # Audio cue markers
                        beepc = 'black'
                        ax[0].axvline(x=audio_stimulus_offset, color=beepc, label='Start audio cue')
                        ax[1].axvline(x=audio_stimulus_offset, color=beepc)
                        
                        # Legend
                        fig.legend(loc='center right', bbox_to_anchor=(0.96, 0.5), fontsize=FS, ncol=1)
                        
                        # Apply font sizes to tick labels
                        for a in ax:
                            a.tick_params(axis='both', which='major', labelsize=FS)
                        
                        plt.tight_layout()
                        plt.subplots_adjust(right=0.8)  # make space for the legend
                        ax[0].grid(False)
                        ax[1].grid(False)
                        plt.savefig('Fig15_rev.pdf', bbox_inches='tight')

        
        
        
        
        
        
        
        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:55:27 2025

@author: joni_
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


FS = 15
alphas = 0.2
ss = 50
bs = 300

object_list = utils.object_list[:]
grasp_data_path = "../results/grasping_data/"

def subsample_classes(x: np.ndarray, y: np.ndarray, k: int, random_state: int = None):
    if random_state is not None:
        np.random.seed(42)

    unique_classes = np.unique(y)
    x_sub, y_sub = [], []

    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        if len(idx) < k:
            raise ValueError(f"Not enough samples for class '{cls}': only {len(idx)} available, requested {k}")
        selected = np.random.choice(idx, size=k, replace=False)
        x_sub.append(x[selected])
        y_sub.append(y[selected])

    x_sub = np.concatenate(x_sub, axis=0)
    y_sub = np.concatenate(y_sub, axis=0)

    return x_sub, y_sub

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, figsize=(8,7))
participants = ['participant_1', 'participant_1', 'participant_2', 'participant_3', 'participant_4', 'participant_5'] 
# FIRST SUBPLOT - Static t-SNE
Xs, ys = [], []
for name_c, name in enumerate(participants):
    filename = name+'.csv'
    results_filename = name+'_results'

    grasp_df = utils.load_dataframe_from_csv(grasp_data_path+name+'_Xy.csv')
    X = grasp_df[utils.glove_data_columns].to_numpy()
    X = utils.preprocess_data_reduction(X, 5)
    y = grasp_df['class'].to_numpy()
    for c, element in enumerate(X):
        Xs.append(element)
        ys.append(y[c])


# num_s = 400
X, y = np.array(Xs), np.array(ys)
unique, counts = np.unique(y, return_counts=True)
min_occurrences = 500#counts.min()
print('obtained data')
perplexity=5
X, y = subsample_classes(X, y, k=min_occurrences, random_state=42)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
X_tsne = tsne.fit_transform(X)
print('fitted')

mean_list = []
for target in np.arange(9):
    indices = np.where(y==target)
    for point in X_tsne[indices]:
        ax1.scatter(point[0], point[1], color=colors[target], alpha=alphas , s=ss)
    mean_list.append(np.mean(X_tsne[indices], axis=0))

mean_list = np.array(mean_list)
for target in np.arange(9):
    ax1.scatter(mean_list[target,0], mean_list[target,1], color=colors[target], alpha=1, s=bs, zorder=1, edgecolors='black', linewidths=1, label=object_list[target])   


# SECOND SUBPLOT - Dynamic t-SNE
data_path = "../../data/dataframes/"
Xs, ys = [], []
for name_c, name in enumerate(participants):#
    filename = name+'.csv'
    df = utils.load_dataframe_from_csv(data_path+"/processed_"+name+'.csv')
    for block in range(0, df['block'].max()+1):
            for trial_number in np.arange(df[(df['block'] == block)]['trial_number'].max()+1):        
                for rep_counter in range(df[(df['block'] == block) & (df['trial_number'] == trial_number)]['rep_number'].max()+1):                                 
                    df_att = df[(df['block'] == block) & (df['trial_number'] == trial_number) & (df['rep_number'] == rep_counter)]
                    if df_att['trial_success'].to_numpy()[-1] == 1:
                        glove_time = df_att['glove_timestamps'].to_numpy()
                        glove_data = df_att[utils.glove_data_columns].to_numpy()
                        board_data = df_att[utils.board_data_columns].to_numpy()
                        board_data = utils.replace_with_highest(arr=board_data, target=8190.0)
                        board_time = df_att['board_timestamps'].to_numpy()
                        palm_data = df_att['palm_data'].to_numpy()
                        palm_time = df_att['palm_timestamps'].to_numpy()
                        y_true = int(df_att['target_object'].to_numpy()[-1])
                        start = np.where(df_att['hand_active'] == 1)[0][0]
                        end = -1
                        board_ts = board_data[start:end, int(trial_number)]
                        board_ts = board_ts - board_ts[0]
                        t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max()*0.1)
                        
                        mov_data = utils.preprocess_data_reduction(glove_data[start:start+t_lift], 5)
                  
                        mov_data = utils.resample(x=mov_data, num=100)
                        Xs.append(mov_data.flatten())
                        ys.append(y_true)

Xs, ys = np.array(Xs), np.array(ys)
X, y = Xs, ys
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
X_tsne = tsne.fit_transform(X)

mean_list = []
for target in np.arange(8):
    indices = np.where(y==target)
    for point in X_tsne[indices]:
        ax2.scatter(point[0], point[1], color=colors[target], alpha=alphas, s=ss)
    mean_list.append(np.mean(X_tsne[indices], axis=0))

mean_list = np.array(mean_list)
for target in np.arange(8):
    ax2.scatter(mean_list[target,0], mean_list[target,1], color=colors[target], alpha=1, s=bs, zorder=1, edgecolors='black', linewidths=1)   

# ax2.set_xlabel("t-SNE 1", fontsize=FS)
# ax2.set_ylabel("t-SNE 2", fontsize=FS)
ax1.set_xticks([])
ax2.set_xticks([])
ax1.set_yticks([])
ax2.set_yticks([])
# ax2.set_title("Dynamic t-SNE")
ax1.grid(False)
ax2.grid(False)

ax1.text(0.02, 0.95, 'Grasp',
         transform=ax1.transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.),
         fontsize=FS)

ax2.text(0.02, 0.95, 'Reach-to-grasp',
         transform=ax2.transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.),
         fontsize=FS)

# Place legend on the right side outside the plots
# fig.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize=FS, col=3)
fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.11),   # 0.5 = center, -0.05 = below
           fontsize=FS,
           ncol=5)  # number of columns in legend

# Adjust layout to make room for the legend
plt.tight_layout()
# plt.subplots_adjust(right=0.85)

# Save the figure
plt.savefig('Fig17_rev.pdf', bbox_inches='tight', dpi=300)

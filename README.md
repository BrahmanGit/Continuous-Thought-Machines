README

# Anticipating Human Hand Grasps for Enhanced Interaction: A Bayesian Approach to Negative Latency in Human-Machine Interfaces

This repository contains the code and data used for the manuscript titled "Anticipating Human Hand Grasps for Enhanced Interaction: A Bayesian Approach to Negative Latency in Human-Machine Interfaces".

## Getting Started

All analyses were performed on a Windows machine. While we cannot guarantee it, there should be no trouble running this code on Linux or Mac.

### Environment Setup

The python packages used for this project are: 

```bash
filterpy==1.4.5
matplotlib==3.10.7
numpy==2.3.4
pandas==2.3.3
scikit_learn==1.7.2
scipy==1.16.2
seaborn==0.13.2
statsmodels==0.14.4
tqdm==4.67.1
```

You can install them using:

```bash
pip install -r requirements.txt
```

The R packages used for the statistical analysis are:

```r
readr
lme4
lmerTest
effectsize
```

You can install them in R using:

```r
install.packages(c("readr", "lme4", "lmerTest", "effectsize"))
```

### Model

The implementation of the Model for Anticipating Grasping Intentions (MAGI) can be found in *sub_models.py*. 

### Data, simulations, and pre-processing

To replicate the experiments detailed in the manuscript, the referenced dataset folder (https://figshare.com/account/articles/30018508) should be copied directly next to the *src/* directory.

The data analysis involves a chain of steps necessary to obtain intermediate results that are then used for further analysis.  

1. To preprocess the raw data, run *post_process_frames.py.* The preprocessed dataframes are saved next to the raw dataframes. Preprocessing involves manual corrections and trimming for the known 5 problematic data without movement sequences recorded outside of the intended experimental reach-2-grasp window. Moreover, spatial rotation of to a given reference frame is applied.

2. To extract static grasp data and participant-specific average movement duration to fit MAGI, run *measure_reach_dur_extract_Xy.py*. Extracted participant-specific files are saved in *results/*, and the average movement durations are printed out.

3. Based on the processed dataframes, run *grid_search_parameter_space.py* to run MAGI on all processed dataframes for experimental blocks B1-B4 using parameter configurations used throughout most analyses described in the manuscript. The performance of MAGI for different parameter configurations are stored in dataframes saved in *results/parameter_space/*. The parametrization yielding optimal performances across the used metrics used throughout the manuscript can be computed using *compute_average_latencies_across_parameters.py*, which stores the parametrization resulting in the lowest latency in the folder *results/parameter_space/* as a json-file. 

4. Run *run_all_evaluations.py* to compute the performance of MAGI across various finger combinations, noise levels and object sets.

5. Based on the predictions stored in *results/predictions/*, *compute_negative_latency.py* needs to be run to evaluate the predictions regarding negative latency. Results are saved in *results/latency.*. The results are stored in *results/fingers*, *results/noise* and *results/objects* respectively. 

6. Performance of neural networks can be obtained by running *evaluate_NN.py*, which stores the predictions of both the CNN and LSTM in *predictions/nn*. 

7. The R script *figures/glm_analysis.R* performs the statistical analysis of negative latency using a linear mixed-effects model. It loads the evaluation results from step 5, filters for the all-fingers condition, computes negative latency, and fits a mixed model with lifted target and validation phase as fixed effects and participant as a random effect. The script outputs coefficient-level results with FDR-corrected p-values as well as an ANOVA table with F-statistics and FDR-corrected p-values.

8. Visualization of all figures is possible by first running steps 1-5 and the running the respective scripts in *figures/*.
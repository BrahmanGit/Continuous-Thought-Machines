It does not yet show:

1. Generalization to unseen trajectories.
2. Prediction from partial trajectories.
3. Stable early prediction.
4. Negative latency.
5. An advantage over a simple classifier.
6. Meaningful confidence calibration.
7. A benefit from CTM internal recurrence or synchronization.

The current training accuracy is measured on the same 1,600 trajectories used to update the weights. This is training-set performance, not validation performance.

synthetic data enough for:
* Testing tensor dimensions
* Debugging the CTM interface
* Confirming gradients flow
* Checking MPS compatibility
* Implementing prefix sampling
* Implementing top-k predictions
* Implementing the negative-latency metric
* Testing plotting and evaluation code
* Testing adaptive stopping over internal ticks
* Testing whether shorter prefixes can be processed

not enough for:
* Claiming the CTM anticipates human intent
* Claiming biological or kinematic realism
* Comparing fairly with MAGI
* Claiming robustness to inter-person variability
* Claiming spatial generalization
* Claiming reliable early anticipation under ambiguity

The MAGI paper defines k-negative latency using the earliest external timestamp at which the correct target enters the top-k set and remains there continuously until the object-lift time.
Therefore, my next experiment must evaluate prefixes such as:
    first 5 frames
    first 10 frames
    first 15 frames
    ...
    first 100 frames

The next immediate modification should be a train-validation split. becuase:
Our current loader shuffles the entire dataset and trains on all of it. There is no held-out validation set.
Therefore, perfect accuracy could mean either:
* The model learned a general class rule.
* The model memorized the finite dataset.
* Some combination of both.
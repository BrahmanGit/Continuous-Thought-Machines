"""
Created on Wed Apr 24 12:18:55 2024

@author: jonas
"""
import logging
from tqdm import tqdm
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import scipy
from scipy.stats import dirichlet
import copy
from sklearn.neighbors import NearestCentroid
from typing import Union, List
from numba import jit, prange
import time


def mean_average_filter(signal, N):
    """
    Applies a mean average filter of length N to a 1D signal while preserving its length.
    
    Parameters:
    signal (array-like): The input signal.
    N (int): The length of the moving average filter.
    
    Returns:
    np.ndarray: The filtered signal with the same length as the input.
    """
    if N <= 0:
        raise ValueError("Filter length N must be positive.")
    
    if not isinstance(N, int):
        raise ValueError("Filter length N must be an integer.")
    
    filtered = np.convolve(signal, np.ones(N)/N, mode='same')
    return filtered


def nearest_positive_definite(A):
    """
    Find the nearest positive-definite matrix to input matrix A.

    This function implements the algorithm from Higham (1988) to find the
    nearest positive-definite matrix in the Frobenius norm.

    Parameters
    ----------
    A : ndarray
        Input matrix, which may not be positive-definite

    Returns
    -------
    A3 : ndarray
        Nearest positive-definite matrix to A
    """
    try:
        B = (A + A.T) / 2  # Ensure symmetry
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        # Add small diagonal if needed
        if not is_positive_definite(A3):
            spacing = np.spacing(np.linalg.norm(A))
            I = np.eye(A.shape[0])
            k = 1
            while not is_positive_definite(A3):
                A3 += I * spacing * k**2
                k += 1
        return A3

    except BaseException:
        return A


def is_positive_definite(A):
    """
    Check if a matrix is positive-definite.

    A matrix is positive-definite if all its eigenvalues are positive.
    This function uses Cholesky decomposition as an efficient test,
    which only succeeds for positive-definite matrices.

    Parameters
    ----------
    A : ndarray
        Square matrix to check

    Returns
    -------
    bool
        True if matrix is positive-definite, False otherwise
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def remove_outlier_rows(
        data: Union[List[List[float]], np.ndarray], threshold: float = 1.5) -> np.ndarray:
    """
    Remove outlier rows from a dataset using the IQR method.

    Identifies and removes rows that contain values outside the range:
    [Q1 - threshold*IQR, Q3 + threshold*IQR] for any column.

    Parameters
    ----------
    data : Union[List[List[float]], np.ndarray]
       Input data, either as a list of lists or numpy array
    threshold : float, optional
       Multiplier for IQR to determine outlier boundaries, default 1.5

    Returns
    -------
    np.ndarray
       Filtered data with outlier rows removed

    Notes
    -----
    This function applies the outlier detection independently to each column,
    and removes any row that has at least one outlier value in any column.
    """
    data = np.array(data, dtype=float)
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    return np.array(data[mask])


def get_likelihood_kf(kf, observation):
    """Return the marginal loglikelihood.

    Based on:
    https://en.wikipedia.org/wiki/Kalman_filter#Marginal_likelihood
    """
    ll = -0.5 * (kf.y.T.dot(np.linalg.inv(kf.S)).dot(kf.y) +
                 np.log(np.linalg.det(kf.S)))
    return ll


def postprocess_likelihoods(ll):
    """Do this special call to softmax to deceive, inveigle and obfuscate."""
    return scipy.special.softmax(ll)


class SimpleAttractor:
    """Dynamics in the form of a simple attractor."""

    def __init__(self, x0, dt, **dyn_pars):
        """Set up the attractor.

        Parameters
        ----------
        x0 : ndarray
        Initial state. For now, used only for inferring the sizes of the space.

        dt : float
        Size of time step.

        **dyn_pars are assumed to be all model parameters, consistent with the
        x_dot method. In this case:

        beta : float
        Coefficient on the RHS of the dynamical system.

        center_point : ndarray
        Center of the gravitational well. Its size must be consistent with that
        of x0.

        """
        self.dyn_pars = dyn_pars
        self.dt = dt  # Set the time step
        self.x = x0  # Set the initial state
        self.z = x0  # Placeholder for observations
        self.x_dim = x0.shape[0]
        self.z_dim = self.x_dim  # placeholder for observations size

    def x_dot(self, x):
        """Return the RHS of the dynamical system."""
        return self.dyn_pars['beta'] * (self.dyn_pars['center'] - x)

    def run_dynamics_dt(self, x, dt):
        """Return the t+1 state of the system."""
        return x + self.x_dot(x) * dt

    def observe_dynamics(self, x):
        """Return observations given the state --x--."""
        return x


class ConstructUKF(UKF):
    """Class to create an unscented kalman filter.

    Based on:
    https://pykalman.github.io/

    Dynamics contains an object of class GravitationalSystem.

    No new methods nor attributes are defined.
    """

    def __init__(self, dynamics, x_init, dt, noise_KF, noise_obs):
        """Initialize an UKF with the given dynamics and noises.

        Convenience class to initiate the UKF with our dynamics.

        TODO: Finish docstring.

        Parameters
        ----------
        noise_KF : float or ndarray
        If float, the dynamics covariance matrix is taken to be diagonal with
        elements equal noise_KF. If ndarray, it is assumed to be the covariance
        matrix itself.

        noise_obs : float or ndarray
        If float, the observation covariance matrix is taken to be diagonal
        with elements equal noise_KF. If ndarray, it is assumed to be the
        covariance matrix itself.

        """
        self.dim_x = x_init.shape[0]  # dim_x
        z_sample = dynamics.observe_dynamics(x_init)
        self.dim_z = z_sample.shape[0]  # dim_z
        self.dynamics = dynamics
        self.x_init = x_init
        self.dt = dt
        self.noise_KF = noise_KF
        self.noise_obs = noise_obs
        # self.X = X  # TODO: Where was this used?

        self.dynamics.x = x_init
        f_cv = self.dynamics.run_dynamics_dt
        h_cv = self.dynamics.observe_dynamics

        # Generate sigma points for the Unscented Kalman Filter
        points = MerweScaledSigmaPoints(
            n=self.dim_x, alpha=1, beta=2, kappa=3 - self.dim_x)
        # points = JulierSigmaPoints(n=self.dim_x, kappa=1)

        super().__init__(dim_x=self.dim_x, dim_z=self.dim_z, fx=f_cv, hx=h_cv,
                         dt=self.dt, points=points)

        # Set initial state and covariance matrices for the Kalman filter
        self.x = self.x_init
        # Measurement noise covariance:
        if isinstance(self.noise_obs, np.ndarray):
            self.R = self.noise_obs
        else:
            self.R = np.eye(self.x_init.shape[0]) * self.noise_obs
        # Process noise covariance:
        if isinstance(self.noise_KF, np.ndarray):
            self.Q = self.noise_KF
        else:
            self.Q = np.eye(self.x_init.shape[0]) * self.noise_KF
        # Initial state covariance:
        self.P = np.eye(self.x_init.shape[0])
        self.initial_state_mean = self.x_init

    def run_inference(self, observation):
        """Run the update function according to pykalman.

        DEPRECATED. Use self.update(observation) instead.
        """
        self.observation = observation
        self.update(observation)

    def get_prediction(self, observation=0):
        """Run predict according to pykalman.

        DEPRECATED. Use self.predict instead.
        """
        self.predict()
        self.observation = observation
        return self.x

    def get_state(self):
        """Return current stae of the system."""
        return self.x


class DirichletModel:
    """Bayesian model selection class based on Dirichlet priors.

    Mainly used class for prediction of goal states (e.g. gestures). Composed
    models containes multiple BaseModels (base_model_lst) that are iterated
    through for:

    -inference (updating the individual models)
    -returning the likelihood of each model (get_likelihood_windo)
    -accumulating the likelihoods (get_accumulated_evidence)
    -obtaining predictions (predict)
    -predicteing the most likely goal states based on the evidence via the
     dirichlet distribution

    BaseModels are reset after reset_interval.

    """

    def __init__(self, base_model_lst, integration_window_length,
                 speed_model_lst=None):
        """TODO: Write the docstring."""
        self.base_model_lst = base_model_lst
        self.model_lst_for_reset = copy.deepcopy(base_model_lst)
        self.speed_model_lst = speed_model_lst
        self.integration_window_length = integration_window_length
        self.epsilon = 1e-15
        self.integration_window = np.ones(
            (integration_window_length, len(base_model_lst))) * self.epsilon
        self.kf_reset_interval = 10
        self.kf_reset_counter = 0

    def evidence_integration(self, evidence_window):
        """Accumulate evidence via simple summation."""
        summed = np.sum(evidence_window, axis=0)
        return summed

    def run_inference(self, observation, t_obs):
        """Run the update step for each internal model."""
        for model_c, model in enumerate(self.base_model_lst):
            self.base_model_lst[model_c].internal_model.predict()
            self.base_model_lst[model_c].internal_model.update(observation)
  
    def update_likelihood_window(self,):
        """Update the likelihood window.

        The likelihood is updated with the latest likelihoods in a FIFO-Buffer
        fashion

        """
        num_mods = len(self.base_model_lst)
        likelihoods_ukf = np.ones(num_mods)
        for ix_model, model in enumerate(self.base_model_lst):
            # likelihoods_ukf[ix_model] = model.likelihood
            likelihoods_ukf[ix_model] = self.base_model_lst[ix_model].internal_model.likelihood

        likelihoods_speed = np.ones(num_mods)
        if self.speed_model_lst is not None:
            for ix_smodel, smodel in enumerate(self.speed_model_lst):
                likelihoods_speed[ix_smodel] = smodel.likelihood

        likelihoods = (likelihoods_ukf * likelihoods_speed)[None, :]
        self.integration_window = np.append(
            self.integration_window, likelihoods, axis=0)[1:]

    def get_accumulated_evidence(self):
        """Return the evidence.

        Returns the evidence, which is the likelihood_window integrated via
        evidence integration (e.g. simple summation, weighted summation)

        """
        return self.evidence_integration(self.integration_window)

    def reset_kalmans(self, x_init):
        """Reset the internal Kalman filters.

        Parameters
        ----------
        x_init : ndarray
        Values for the initial state of the Kalman filters.

        """
        # print('reset')
        reset_models = copy.deepcopy(self.model_lst_for_reset)
        for ix_model, model in enumerate(reset_models):
            
            curr_dyn = model.dynamics
            
            # x0 = x_init #resetting using last observation
            x0 = self.base_model_lst[ix_model].x
            dyn = SimpleAttractor(center=curr_dyn.dyn_pars['center'], x0=x0, dt=curr_dyn.dt,
                                  beta=curr_dyn.dyn_pars['beta'])
            
            kf = ConstructUKF(dynamics=dyn, x_init=x0, dt=model.dt,
                              noise_KF=model.noise_KF, noise_obs=model.noise_obs)
            
            self.base_model_lst[ix_model].internal_model = kf
       

    def update_posteriors(self,):
        """Update the likelihood window and return the accumulated evidence."""
        self.update_likelihood_window()
        log_acc_evidence = self.get_accumulated_evidence() 
        return log_acc_evidence

    def run_sequence(self, observations, t_obs):
        """Run all inference and update functions for all observations.
    
        Parameters
        ----------
        observations : ndarray, size=(N, M)
            N observations in an M-dimensional space.
        """
        y_pred_mean = []
        y_pred_var = []
        ll = []
    
        for obs, t in tqdm(zip(observations, t_obs), total=len(observations), desc="Running inference"):
            if self.kf_reset_counter % self.kf_reset_interval == 0:
                self.reset_kalmans(x_init=obs)  # TODO: This should be state, not obs.
            self.run_inference(obs, t)
            acc_evidence = self.update_posteriors()
            nacc_evidence = acc_evidence / acc_evidence.max() #TODO: removed because of numerical issues
            if 0 in nacc_evidence: nacc_evidence += + np.ones(len(nacc_evidence))*self.epsilon 
            y_pred_mean.append(dirichlet.mean(nacc_evidence))
            y_pred_var.append(dirichlet.var(nacc_evidence))
            ll.append(acc_evidence)
            self.kf_reset_counter += 1
    
        return np.array(y_pred_mean), np.array(y_pred_var), np.array(ll)

    def run_single_step(self, obs, t_obs):
        if self.kf_reset_counter % self.kf_reset_interval == 0:
            self.reset_kalmans(x_init=obs) 
        self.run_inference(obs, t_obs)
        acc_evidence = self.update_posteriors()
        nacc_evidence = acc_evidence / acc_evidence.max()
        if 0 in nacc_evidence: nacc_evidence += + np.ones(len(nacc_evidence))*self.epsilon 
        mean = dirichlet.mean(nacc_evidence)
        var = dirichlet.var(nacc_evidence)
        ll = acc_evidence
        self.kf_reset_counter += 1
        return mean, var, ll

    def predict(self, observation, t_obs):
            obs = observation
            if self.kf_reset_counter % self.kf_reset_interval == 0:
                self.reset_kalmans(obs) 
            # self.run_inference(obs, t_obs)
            acc_evidence = self.update_posteriors()
            nacc_evidence = acc_evidence / acc_evidence.max()
            self.kf_reset_counter += 1
            return dirichlet.mean(nacc_evidence), dirichlet.var(nacc_evidence), acc_evidence
    

class GravitationalPredictive(DirichletModel):
    """Dirichlet predictive model initialized with attractor dynamics."""

    def __init__(self, centers, noise_KF, noise_obs, x0, dt,
                 beta, **kwargs):
        """
        
        Parameters
        ----------
        centers : ndarray, size(N, M)
        Centers for the attractor dynamics (e.g. gestures). N centers of
        dimension M each.
        """
        model_lst = []
        kf_constructor_lst = []
        if isinstance(noise_KF, np.ndarray):
            if np.ndim(noise_KF) == 2:
                noise_KF = np.tile(noise_KF[None, :, :], (len(centers), 1, 1))
        else:
            noise_KF = noise_KF * np.ones(len(centers))
        if isinstance(noise_obs, np.ndarray):
            if np.ndim(noise_obs) == 2:
                noise_KF = np.tile(noise_obs[None, :, :], (len(centers), 1, 1))
        else:
            noise_obs = noise_obs * np.ones(len(centers))
        for nkf, nobs, center in zip(noise_KF, noise_obs, centers):
            dyn = SimpleAttractor(center=center, x0=x0, dt=dt,
                                  beta=beta)

            kf = ConstructUKF(dynamics=dyn, x_init=x0, dt=dt,
                              noise_KF=nkf, noise_obs=nobs)
            model_lst.append(kf)
            kf_constructor_lst.append(kf)
        super().__init__(base_model_lst=model_lst,
                         # kf_constructor_lst=kf_constructor_lst,
                         **kwargs)


class GravFromData(GravitationalPredictive):
    """Example usage of the parent class.

    Extracts the centers for the attractors by using NearestCentroid for
    clustering multiple observations of the gestures in the data.
    """

    def __init__(self, observed_gestures, labels, *args, **kwargs):
        """Obtain the centroids and initialize the Dirichlet model.

        Parameters
        ----------
        observed_gestures : ndarray, size=(N, M)
        N observations from data, in an M-dimensional space. For example, N
        gestures observed with M sensors.

        labels : ndarray[int], size=(N,)
        Labels for each one of the N observations.

        """
        centroids = self.extract_centroids(observed_gestures, labels)
        super().__init__(centroids, *args, **kwargs)

    @staticmethod
    def extract_centroids(points, labels, centroid_fun=None):
        """Extract the centroids from the --points--.

        TODO: Finish documentation.
        """
        if centroid_fun is None:
            logging.info('Choosing default centroid_fun')
            centroid_fun = NearestCentroid()
        clf = centroid_fun
        clf.fit(points, labels)
        cp_lst = clf.centroids_
        return cp_lst

class NearestCentroidClassifier():
    """
    Nearest Centroid Classifier
    
    A simple classifier that assigns to an observation the class of the nearest centroid.
    This classifier is similar to K-means but is used for classification rather than clustering.
    The predictions are accumulated over the last self.integration window steps. 
    """
    
    def __init__(self):
        self.centroids = None
        self.classes = None
        self.integration_window_length = 10
    
    def fit(self, X, y):
        
        X = np.array(X)
        y = np.array(y)
        
        # Get unique classes
        self.classes = np.unique(y)
        
        # Calculate centroids for each class
        self.centroids = {}
        for cls in self.classes:
            # Select all points that belong to the current class
            X_cls = X[y == cls]
            # Calculate the mean of these points (this is the centroid)
            self.centroids[cls] = np.mean(X_cls, axis=0)
            
        return self
    
    def predict(self, X):
        
        X = np.array(X)
        predictions = []
        
        for sample in X:
            # Calculate distance to each centroid
            distances = {}
            for cls, centroid in self.centroids.items():
                # Euclidean distance
                distances[cls] = np.sqrt(np.sum((sample - centroid) ** 2))
            
            # Find the class with the minimum distance
            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)
            
        return np.array(predictions)
    
    
    def one_hot_encode(self, array):
        array = np.array(array)
        unique_values = np.unique(self.classes)       
        value_to_index = {value: i for i, value in enumerate(unique_values)}       
        one_hot = []   
        # Fill in the one-hot encoded array
        for value in array:
            vector = [0] * len(unique_values)         
            vector[value_to_index[value]] = 1          
            one_hot.append(vector)
        
        return one_hot
    
    def accumulate_evidence(self, pred_window):
        if len(pred_window) >= 1:
            return np.mean(np.array(pred_window), axis=0)
        else: return np.array(pred_window) 
    
    def run_sequence(self, seq):
        y_pred = self.predict(seq)
        y_pred_oh = self.one_hot_encode(y_pred)
        pred_win = np.zeros((seq.shape[0], *self.accumulate_evidence(y_pred_oh[0:1]).shape))
        window_size = self.integration_window_length
        
        for i in range(seq.shape[0]):
            start_idx = max(0, i - window_size)
            window = y_pred_oh[start_idx:i+1]
            pred_win[i] = self.accumulate_evidence(window)
        
        return pred_win

    def accumulate_evidence_dirichlet(self, pred_window):
        epsilon = 1e-200
        pred_window = np.array(pred_window)
        if pred_window.ndim == 2:
            alpha = np.ones(pred_window.shape[1])*epsilon
            for j in range(len(pred_window)):
                alpha[np.argmax(pred_window[j])] += 1
            return dirichlet.mean(alpha)
                
        else: 
            alpha = np.ones(pred_window.shape[0])*epsilon
            alpha[np.argmax(pred_window[j])] = 1
            return dirichlet.mean(alpha)

    def run_sequence_proba(self, seq):
        y_pred = self.predict(seq)
        y_pred_oh = self.one_hot_encode(y_pred)
        pred_win = np.zeros((seq.shape[0], *self.accumulate_evidence(y_pred_oh[0:1]).shape))
        window_size = self.integration_window_length
        
        for i in range(seq.shape[0]):
            start_idx = max(0, i - window_size)
            window = y_pred_oh[start_idx:i+1]
            pred_win[i] = self.accumulate_evidence_dirichlet(window)
        
        return pred_win



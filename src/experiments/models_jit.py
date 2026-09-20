import numpy as np
import copy
from tqdm import tqdm
from scipy.stats import dirichlet
from sklearn.neighbors import NearestCentroid
from numba import jit, prange

# ============= CONFIGURATION =============
USE_NUMBA = True  # Set to False to disable numba jitting
# =========================================

def conditional_jit(*args, **kwargs):
    """Decorator that applies jit only if USE_NUMBA is True"""
    def decorator(func):
        if USE_NUMBA:
            return jit(*args, **kwargs)(func)
        return func
    return decorator


def multivariate_normal_pdf(x, mean, cov):
    """
    Multivariate normal PDF implemented with NumPy.
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if mean.ndim > 1:
        mean = mean.flatten()

    n, d = x.shape

    det_cov = np.linalg.det(cov)
    if det_cov <= 0:
        raise ValueError("Covariance matrix must be positive definite")
    inv_cov = np.linalg.inv(cov)

    norm_const = np.power((2 * np.pi), -d / 2) * np.power(det_cov, -0.5)

    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

    pdf = norm_const * np.exp(exponent)
    return pdf if n > 1 else pdf.item()


@conditional_jit(nopython=True)
def _symmetrize(M):
    return 0.5 * (M + M.T)


@conditional_jit(nopython=True)
def _generate_sigma_points(mean, covariance, dim_x, lambda_, noise_cov):
    """Generate sigma points for the unscented transform."""
    n = dim_x
    k = lambda_
    scaled_P = (n + k) * covariance
    evals, evecs = np.linalg.eigh(_symmetrize(scaled_P))
    evals = np.maximum(evals, 0.0)
    L = evecs @ np.diag(np.sqrt(evals))

    sigma = np.zeros((2 * n + 1, n))
    sigma[0] = mean
    for i in range(n):
        col = L[:, i]
        sigma[1 + i] = mean + col
        sigma[1 + n + i] = mean - col
    return sigma


@conditional_jit(nopython=True)
def fx(x, center, grav, dt):
    """Attractor dynamics: gravitational pull towards center."""
    return x + grav * (center - x) * dt


@conditional_jit(nopython=True)
def _unscented_transform(sigma_points, dt, Wm, Wc, Q, center, grav):
    """Apply unscented transform for prediction step."""
    n_sigma, state_dim = sigma_points.shape
    transformed_points = np.empty_like(sigma_points)

    for i in range(n_sigma):
        transformed_points[i] = fx(sigma_points[i], center, grav, dt)

    predicted_state_mean = np.sum(Wm.reshape(-1, 1) * transformed_points, axis=0)

    predicted_state_covariance = np.zeros((state_dim, state_dim))
    for i in range(n_sigma):
        wc = Wc[i]
        for a in range(state_dim):
            diff_a = transformed_points[i, a] - predicted_state_mean[a]
            for b in range(state_dim):
                diff_b = transformed_points[i, b] - predicted_state_mean[b]
                predicted_state_covariance[a, b] += wc * diff_a * diff_b

    predicted_state_covariance += Q
    return predicted_state_mean, predicted_state_covariance, transformed_points


@conditional_jit(nopython=True)
def _unscented_observation(sigma_points, Wm, Wc, R):
    """Apply unscented transform for observation step."""
    n_sigma, state_dim = sigma_points.shape
    transformed_points = sigma_points

    mean = np.sum(Wm.reshape(-1, 1) * transformed_points, axis=0)

    covariance = np.zeros((state_dim, state_dim))
    for i in range(n_sigma):
        wc = Wc[i]
        for a in range(state_dim):
            diff_a = transformed_points[i, a] - mean[a]
            for b in range(state_dim):
                diff_b = transformed_points[i, b] - mean[b]
                covariance[a, b] += wc * diff_a * diff_b

    covariance += R
    return mean, covariance, transformed_points


class FastUnscentedKalmanFilter:
    """Fast implementation of Unscented Kalman Filter with attractor dynamics."""

    def __init__(self, dim_x, dim_z, dt, center, grav):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.alpha = 0.1
        self.beta = 2.0
        self.kappa = 3 - self.dim_x
        self.dt = dt

        # Noise matrices (to be set externally)
        self.Q = None  # Process noise
        self.R = None  # Observation noise
        self.P = None  # State covariance
        self.x = None  # Current state

        # Attractor parameters
        self.center = center
        self.grav = grav

        # Predicted quantities
        self.predicted_state_mean = None
        self.predicted_state_covariance = None
        self.predicted_obs_mean = None
        self.predicted_obs_cov = None

        # UKF parameters
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.n_sigma = 2 * dim_x + 1

        # Weights
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)

        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)

        for i in range(1, self.n_sigma):
            self.Wm[i] = self.Wc[i] = 1.0 / (2 * (dim_x + self.lambda_))

        self.noise_cov = 1e-7
        self.likelihood = None

    def predict(self):
        sigma_points = _generate_sigma_points(self.x, self.P, self.dim_x, self.lambda_, self.noise_cov)
        self.predicted_state_mean, self.predicted_state_covariance, _ = _unscented_transform(
            sigma_points, self.dt, self.Wm, self.Wc, self.Q, self.center, self.grav
        )
        self.predicted_state_covariance = _symmetrize(self.predicted_state_covariance) + np.eye(self.dim_x) * self.noise_cov
        

    def update(self, observation):
        observation = np.asarray(observation)

        sigma_points = _generate_sigma_points(
            self.predicted_state_mean, self.predicted_state_covariance, 
            self.dim_x, self.lambda_, self.noise_cov
        )
        

        self.predicted_obs_mean, self.predicted_obs_cov, transformed_sigma_points = _unscented_observation(
            sigma_points, self.Wm, self.Wc, self.R
        )

        state_diff = sigma_points - self.predicted_state_mean
        obs_diff = transformed_sigma_points - self.predicted_obs_mean
        cross_covariance = np.sum(
            self.Wc.reshape(-1, 1, 1) * 
            np.array([np.outer(sd, od) for sd, od in zip(state_diff, obs_diff)]), 
            axis=0
        )

        try:
            kalman_gain = cross_covariance @ np.linalg.inv(self.predicted_obs_cov)
        except np.linalg.LinAlgError:
            kalman_gain = cross_covariance @ np.linalg.pinv(self.predicted_obs_cov)

        innovation = observation - self.predicted_obs_mean
        self.x = self.predicted_state_mean + kalman_gain @ innovation
        self.P = (self.predicted_state_covariance - 
                 kalman_gain @ self.predicted_obs_cov @ kalman_gain.T)

        self.likelihood = multivariate_normal_pdf(
            observation, self.predicted_obs_mean, self.predicted_obs_cov
        )


# ------------------ Parallel numba functions (used by Dirichlet model) ------------------
@conditional_jit()
def _compute_likelihood(x, mean, cov):
    d = len(mean)
    diff = x - mean

    det_cov = np.linalg.det(cov)
    if det_cov <= 0:
        return 1e-15

    inv_cov = np.linalg.inv(cov)
    norm_const = np.power((2 * np.pi), -d / 2) * np.power(det_cov, -0.5)
    exponent = -0.5 * np.sum(diff * (inv_cov @ diff))

    return norm_const * np.exp(exponent)


@conditional_jit(parallel=True)
def _parallel_predict_update(states, covariances, centers, observations, 
                           Wm_array, Wc_array, Q_array, R_array, dt, grav, 
                           alpha, beta, kappa, noise_cov):
    n_models, dim_x = states.shape
    lambda_ = alpha**2 * (dim_x + kappa) - dim_x

    new_states = np.empty_like(states)
    new_covariances = np.empty_like(covariances)
    likelihoods = np.empty(n_models)

    for i in prange(n_models):
        x = states[i]
        P = covariances[i]
        center = centers[i]
        Wm = Wm_array[i]
        Wc = Wc_array[i]
        Q = Q_array[i]
        R = R_array[i]

        sigma_points = _generate_sigma_points(x, P, dim_x, lambda_, noise_cov)

        predicted_state_mean, predicted_state_covariance, _ = _unscented_transform(
            sigma_points, dt, Wm, Wc, Q, center, grav
        )
        predicted_state_covariance = _symmetrize(predicted_state_covariance) + np.eye(dim_x) * noise_cov

        sigma_points_pred = _generate_sigma_points(
            predicted_state_mean, predicted_state_covariance, dim_x, lambda_, noise_cov
        )

        predicted_obs_mean, predicted_obs_cov, transformed_sigma_points = _unscented_observation(
            sigma_points_pred, Wm, Wc, R
        )

        state_diff = sigma_points_pred - predicted_state_mean
        obs_diff = transformed_sigma_points - predicted_obs_mean

        cross_covariance = np.zeros((dim_x, dim_x))
        for j in range(len(Wc)):
            wc = Wc[j]
            for a in range(dim_x):
                for b in range(dim_x):
                    cross_covariance[a, b] += wc * state_diff[j, a] * obs_diff[j, b]

        kalman_gain = cross_covariance @ np.linalg.inv(predicted_obs_cov)

        innovation = observations - predicted_obs_mean

        new_states[i] = predicted_state_mean + kalman_gain @ innovation
        new_covariances[i] = (predicted_state_covariance - 
                             kalman_gain @ predicted_obs_cov @ kalman_gain.T)

        likelihoods[i] = _compute_likelihood(observations, predicted_obs_mean, predicted_obs_cov)

    return new_states, new_covariances, likelihoods


@conditional_jit(parallel=True)
def _parallel_reset_models(n_models, dim_x, x0):
    states = np.empty((n_models, dim_x))
    covariances = np.empty((n_models, dim_x, dim_x))

    for i in prange(n_models):
        states[i] = x0
        covariances[i] = np.eye(dim_x)

    return states, covariances


# ------------------ Dirichlet model using the parallel functions ------------------
class DirichletKalmanModel:
    """
    Bayesian model selection using Dirichlet priors with multiple UKF models.
    This implementation packs the per-model parameters into arrays and runs the
    predict+update steps in parallel using numba-jitted functions for speed.
    """

    def __init__(self, centers, x0, dt, noise_process, noise_obs, gravitational_const, 
                 integration_window_length, reset_interval):
        self.centers = np.asarray(centers)
        self.x0 = np.asarray(x0)
        self.dt = dt
        self.gravitational_const = gravitational_const
        self.integration_window_length = integration_window_length
        self.reset_interval = reset_interval
        self.reset_counter = 0
        self.epsilon = 1e-15

        # Create a prototype UKF to extract static parameters
        n_models = self.centers.shape[0]
        dim_x = self.x0.shape[0]

        self.base_model_lst = []
        for center in self.centers:
            ukf = FastUnscentedKalmanFilter(
                dim_x=dim_x,
                dim_z=dim_x,
                dt=dt,
                center=center,
                grav=gravitational_const
            )
            ukf.R = np.eye(dim_x) * noise_obs
            ukf.Q = np.eye(dim_x) * noise_process
            ukf.P = np.eye(dim_x)
            ukf.x = self.x0.copy()
            self.base_model_lst.append(ukf)

        # Pack parameters into arrays for numba functions
        self._pack_model_arrays()

        # Evidence window
        self.integration_window = np.ones((integration_window_length, n_models)) * self.epsilon

        # Store original states for reset
        self.original_states = self.states.copy()
        self.original_covariances = self.covariances.copy()

    def _pack_model_arrays(self):
        """Pack attributes from base_model_lst into contiguous numpy arrays for numba."""
        n_models = len(self.base_model_lst)
        dim_x = self.x0.shape[0]

        # states and covariances
        self.states = np.empty((n_models, dim_x))
        self.covariances = np.empty((n_models, dim_x, dim_x))

        # centers
        self.centers = np.empty((n_models, dim_x))

        # weights, Q, R
        n_sigma = 2 * dim_x + 1
        self.Wm_array = np.empty((n_models, n_sigma))
        self.Wc_array = np.empty((n_models, n_sigma))
        self.Q_array = np.empty((n_models, dim_x, dim_x))
        self.R_array = np.empty((n_models, dim_x, dim_x))

        for i, model in enumerate(self.base_model_lst):
            self.states[i] = model.x
            self.covariances[i] = model.P
            self.centers[i] = model.center
            self.Wm_array[i] = model.Wm
            self.Wc_array[i] = model.Wc
            self.Q_array[i] = model.Q
            self.R_array[i] = model.R

    def _unpack_to_models(self, new_states, new_covariances, likelihoods):
        """Unpack arrays produced by numba back into our model objects."""
        for i, model in enumerate(self.base_model_lst):
            model.x = new_states[i]
            model.P = new_covariances[i]
            model.likelihood = likelihoods[i]

    def run_inference(self, observation):
        """Run the parallel predict+update for all models using numba."""
        # Ensure arrays are up-to-date
        # (states/covariances should already be in sync, but keep safe)
        for i, model in enumerate(self.base_model_lst):
            self.states[i] = model.x
            self.covariances[i] = model.P
            # Allow Q/R/weights to be updated externally if needed
            self.Q_array[i] = model.Q
            self.R_array[i] = model.R

        new_states, new_covariances, likelihoods = _parallel_predict_update(
            self.states, self.covariances, self.centers, observation,
            self.Wm_array, self.Wc_array, self.Q_array, self.R_array,
            self.dt, self.gravitational_const,
            self.base_model_lst[0].alpha, self.base_model_lst[0].beta, self.base_model_lst[0].kappa,
            self.base_model_lst[0].noise_cov
        )

        # Update internal arrays and model objects
        self.states = new_states
        self.covariances = new_covariances
        self._unpack_to_models(new_states, new_covariances, likelihoods)

    def update_evidence_window(self):
        likelihoods = np.array([model.likelihood for model in self.base_model_lst])
        likelihoods = likelihoods.reshape(1, -1)
        self.integration_window = np.append(self.integration_window, likelihoods, axis=0)[1:]

    def get_accumulated_evidence(self):
        return np.sum(self.integration_window, axis=0)

    def reset_models(self, x0=None):
        if x0 is None:
            x0 = self.x0
        n_models = self.states.shape[0]
        dim_x = self.states.shape[1]
        states, covariances = _parallel_reset_models(n_models, dim_x, x0)
        self.states = states
        self.covariances = covariances
        # Unpack back into python model objects
        for i, model in enumerate(self.base_model_lst):
            model.x = self.states[i].copy()
            model.P = self.covariances[i].copy()

    def predict_step(self, observation):
        if self.reset_counter % self.reset_interval == 0:
            # reset to observation as initial guess (preserves dimension)
            self.reset_models(observation)

        # Run parallel inference
        self.run_inference(observation)

        # Update evidence window and compute Dirichlet stats
        self.update_evidence_window()
        evidence = self.get_accumulated_evidence()
        max_evidence = evidence.max()
        epsilon = 1e-10
        if max_evidence < epsilon:
            normalized_evidence = np.ones_like(evidence) / len(evidence)
        else:
            normalized_evidence = evidence / (max_evidence + epsilon)
            normalized_evidence = np.maximum(normalized_evidence, epsilon)

        mean = dirichlet.mean(normalized_evidence)
        var = dirichlet.var(normalized_evidence)

        self.reset_counter += 1
        return mean, var, evidence

    def run_sequence(self, observations):
        means, vars, evidence = [], [], []
        for obs in observations:
            mean, var, ev = self.predict_step(obs)
            means.append(mean)
            vars.append(var)
            evidence.append(ev)
        return np.array(means), np.array(vars), np.array(evidence)

    def run_single_step(self, observation):
        return self.predict_step(observation)


# ------------------ Utilities for creating models ------------------
def extract_centroids(X, y, centroid_method=None):
    if centroid_method is None:
        centroid_method = NearestCentroid()
    centroid_method.fit(X, y)
    return centroid_method.centroids_


def create_dirichlet_model(X, y, x0, dt, noise_process=1e-3, noise_obs=1e-3, 
                          gravitational_const=1.0, **kwargs):
    centers = extract_centroids(X, y)
    return DirichletKalmanModel(
        centers=centers,
        x0=x0,
        dt=dt,
        noise_process=noise_process,
        noise_obs=noise_obs,
        gravitational_const=gravitational_const,
        **kwargs
    )
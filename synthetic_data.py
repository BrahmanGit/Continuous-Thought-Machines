import matplotlib.pyplot as plt
import numpy as np
import torch


def make_synthetic_grasps(
    n_per_class=200, n_classes=8, n_channels=15, n_timesteps=100, noise_std=0.05, seed=0
):
    rng = np.random.default_rng(seed)

    rest = rng.standard_normal(n_channels) * 0.1  # one shared start pose
    targets = rng.standard_normal((n_classes, n_channels))  # one end-config per grasp
    ramp = np.linspace(0, 1, n_timesteps)  # 0->1 over the reach

    N = n_classes * n_per_class
    X = np.zeros((N, n_channels, n_timesteps), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)

    idx = 0
    for c in range(n_classes):
        for _ in range(n_per_class):
            # trajectory interpolates rest -> target over 100 steps
            traj = rest[:, None] + (targets[c] - rest)[:, None] * ramp[None, :]
            traj += rng.standard_normal((n_channels, n_timesteps)) * noise_std
            X[idx], y[idx] = traj, c
            idx += 1

    return torch.from_numpy(X), torch.from_numpy(y)


def plot_grasps(X, y, channel=0, classes=(0, 1), n_examples=5):
    plt.figure(figsize=(8, 4))
    for c in classes:
        idxs = (y == c).nonzero(as_tuple=True)[0][:n_examples]
        for k, i in enumerate(idxs):
            plt.plot(
                X[i, channel].numpy(), alpha=0.6, label=f"class {c}" if k == 0 else None
            )
    plt.title(f"Channel {channel}: trajectories per class")
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y = make_synthetic_grasps()
    print(X.shape, y.shape)
    plot_grasps(X, y, channel=0, classes=(0, 3))

import numpy as np


def add_wiener_noise(x, sigma, rng=None, time_axis=-1):
    """
    Add independent Wiener-process noise to each signal channel.

    Parameters
    ----------
    x : np.ndarray
        Input trajectory.
        Expected common shapes:
            (time, channels)
            (samples, time, channels)

    sigma : float
        Wiener noise strength.
        Because time is normalized to [0, 1], sigma approximately
        controls the standard deviation of the noise at the final
        timestep.

    rng : np.random.Generator, optional
        NumPy random-number generator for reproducibility.

    time_axis : int
        Axis corresponding to time.
        For (time, channels) and (samples, time, channels),
        time_axis=-2 is correct.

    Returns
    -------
    x_noisy : np.ndarray
        Trajectory corrupted with Wiener noise.
    """

    x = np.asarray(x, dtype=np.float32)

    if sigma == 0:
        return x.copy()

    if rng is None:
        rng = np.random.default_rng()

    T = x.shape[time_axis]

    if T < 2:
        raise ValueError("Wiener noise requires at least 2 timesteps.")

    # Normalize the complete trajectory duration to [0, 1].
    dt = 1.0 / (T - 1)

    # Brownian increments:
    # dW_t ~ N(0, dt)
    increments_shape = list(x.shape)
    increments_shape[time_axis] = T - 1

    dW = rng.normal(
        loc=0.0,
        scale=np.sqrt(dt),
        size=increments_shape,
    ).astype(np.float32)

    # W_t = cumulative sum of increments
    W = np.cumsum(dW, axis=time_axis)

    # Wiener process starts at W_0 = 0.
    zero_shape = list(x.shape)
    zero_shape[time_axis] = 1

    W0 = np.zeros(zero_shape, dtype=np.float32)
    W = np.concatenate([W0, W], axis=time_axis)

    return x + sigma * W

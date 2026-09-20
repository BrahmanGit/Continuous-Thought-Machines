# -*- coding: utf-8 -*-
"""
Visualization of Bayesian filtering with competing models.

Creates a 2D state space diagram showing:
- Prior distributions for two competing models (m1 and m2)
- Predicted states after applying dynamics
- Observation and likelihood calculations
- Curved arrows showing state transitions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import numpy as np


def curved_arrow(start, end, control, color, label):
    """
    Draw a curved arrow from start to end point using quadratic Bezier curve.
    
    Args:
        start: Starting point [x, y]
        end: Ending point [x, y]
        control: Control point [x, y] that defines the curve shape
        color: Color of the arrow
        label: Label for the legend
    """
    # Define Bezier curve vertices and path codes
    verts = [start, control, end]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    
    # Generate points along the curve using quadratic Bezier formula
    t = np.linspace(0, 1, 50)
    plw = 1  # Line width
    curve_points = []
    for ti in t:
        # Quadratic Bezier: B(t) = (1-t)²*P0 + 2(1-t)t*P1 + t²*P2
        point = (1-ti)**2 * start + 2*(1-ti)*ti * control + ti**2 * end
        curve_points.append(point)
    curve_points = np.array(curve_points)
    
    # Plot the curved dashed line
    ax.plot(curve_points[:, 0], curve_points[:, 1], linestyle='--', 
            color=color, alpha=1, linewidth=plw, label=label)
    
    # Add arrowhead at the end of the curve
    arrow = FancyArrowPatch(curve_points[-5], curve_points[-1], 
                           arrowstyle='->', mutation_scale=30,
                           color=color, linestyle='--', alpha=1, linewidth=plw)
    ax.add_patch(arrow)
    

def draw_orthogonal_line(ax, start, end, color, lw,
                         tick_size=0.15, inset=0.12):
    """
    Draw a line segment with perpendicular tick marks at the endpoint.
    
    Used to visualize likelihood (distance from prediction to observation)
    with perpendicular end caps indicating orthogonal measurement.
    
    Args:
        ax: Matplotlib axis object
        start: Starting point [x, y]
        end: Ending point [x, y]
        color: Line color
        lw: Line width
        tick_size: Half-length of the perpendicular tick mark
        inset: Distance to move the tick inward from endpoint
    """
    # Plot main line segment
    ax.plot([start[0], end[0]], [start[1], end[1]],
            '-', color=color, linewidth=lw)

    # Calculate direction vector
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = np.hypot(dx, dy)
    if length == 0:
        return
    
    # Normalize direction vector and compute perpendicular (90° rotation)
    ux, uy = dx/length, dy/length
    nx, ny = -uy, ux   # Perpendicular vector
    
    # Calculate tick position (slightly inset from endpoint)
    cap_x = end[0] - ux*inset
    cap_y = end[1] - uy*inset
    
    # Draw perpendicular tick mark
    ax.plot([cap_x - nx*tick_size, cap_x + nx*tick_size],
            [cap_y - ny*tick_size, cap_y + ny*tick_size],
            color=color, linewidth=lw)
    

def plot_marker_with_diagonals(ax, xy, size=10, color='k', lw=1, zorder=20, label=None):
    """
    Plot a circular marker with diagonal cross lines (unused in current plot).
    
    Args:
        ax: Matplotlib axis object
        xy: Position [x, y]
        size: Marker size
        color: Marker color
        lw: Line width for diagonals
        zorder: Drawing order (higher = on top)
        label: Legend label
    """
    x, y = xy
    
    # Draw main circle marker
    ax.plot(x, y, 'o', markersize=size, color=color, zorder=zorder, label=label)
    
    # Scale diagonal lines relative to marker size
    diag = size * 0.02  
    
    # Draw X-shaped diagonal lines through marker
    ax.plot([x - diag, x + diag], [y - diag, y + diag], color=color, lw=lw, zorder=zorder+1)
    ax.plot([x - diag, x + diag], [y + diag, y - diag], color=color, lw=lw, zorder=zorder+1)


# === Plot configuration ===
# Define reference dimensions
REFERENCE_WIDTH = 8  # inches
REFERENCE_FONTSIZE = 15

# For each script, calculate scaled font size
fig_width = 8  # your actual figure width
FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)

fig, ax = plt.subplots(figsize=(fig_width, 5))

# Define color scheme for two competing models
mypink = '#416897'   # Blue color for model 1
mygreen = '#EDA985'  # Orange/coral color for model 2

# === Define key points in state space ===
scale = 10  # Marker size
lw = 1      # Line width

# Prior state estimates at time τ-1
pinit1 = np.array([0, 0.8])    # Prior for model 1
pinit2 = np.array([1.1, 0.5])  # Prior for model 2

# Plot prior state estimates
ax.plot(*pinit1, 'o', color=mypink, markersize=scale, zorder=1, 
        label=r"$p(\hat{x}_{\tau-1}|m_1)$")
ax.plot(*pinit2, 'o', color=mygreen, markersize=scale, zorder=1, 
        label=r"$p(\hat{x}_{\tau-1}|m_2)$")

# Uncertainty ellipses around prior estimates
init_ellipse1 = patches.Ellipse(pinit1, 1.4, 1.0, angle=30, fill=True, 
                                facecolor='white', edgecolor=mypink, 
                                alpha=0.7, zorder=-1)
ax.add_patch(init_ellipse1)

init_ellipse2 = patches.Ellipse(pinit2, 0.6, 0.8, angle=-80, fill=True, 
                                facecolor='white', edgecolor=mygreen, 
                                alpha=0.7, zorder=-1)
ax.add_patch(init_ellipse2)

# Predicted states at time τ (after applying dynamics f(·,m))
p1 = np.array([0.8, 2.5])   # Prediction from model 1
p2 = np.array([4.1, 2.4])   # Prediction from model 2

# Actual observation at time τ
obs = np.array([1.3, 3.5])

# === Draw prediction uncertainty ellipses ===
plw = 0.5  # Unused variable (can be removed)

# Prediction uncertainty for model 1 (pink/blue)
pink_ellipse = patches.Ellipse(p1, 2.8, 1.8, angle=-50, fill=True, 
                               facecolor=mypink, edgecolor=mypink, 
                               alpha=0.2, zorder=-1, linestyle='-')
ax.add_patch(pink_ellipse)

# Prediction uncertainty for model 2 (green/orange)
green_ellipse = patches.Ellipse(p2, 3.6, 4.5, angle=165, fill=True, 
                                facecolor=mygreen, edgecolor=mygreen, 
                                alpha=0.2, zorder=-1, linestyle='-')
ax.add_patch(green_ellipse)

# === Plot predicted states and observation ===
# Predicted state for model 1
ax.plot(*p1, 'o', color='white', markersize=scale, zorder=20, 
        markeredgecolor=mypink, 
        label=r"$p(\tilde{x}_{\tau} \mid \hat{x}_{\tau-1}, m_1)$")

# Predicted state for model 2
ax.plot(*p2, 'o', color='white', markersize=scale, zorder=20, 
        markeredgecolor=mygreen, 
        label=r"$p(\tilde{x}_{\tau} \mid \hat{x}_{\tau-1}, m_2)$")

# Actual observation
ax.plot(*obs, 'ko', label=r'$o_{\tau}$', color='white', markersize=scale, 
        zorder=20, markeredgecolor='black', markeredgewidth=2.0)

# === Draw state transitions (curved arrows) ===
# Arrow showing dynamics f(·,m1): prior → prediction for model 1
curved_arrow(pinit1, p1, np.array([0.9, 1.3]), mypink, r'$f(\cdot,m_1)$')

# Arrow showing dynamics f(·,m2): prior → prediction for model 2
curved_arrow(pinit2, p2, np.array([2.0, 2.5]), mygreen, r'$f(\cdot,m_2)$')

# === Draw likelihood lines (distance from prediction to observation) ===
# Likelihood for model 1: distance between p1 prediction and observation
ax.plot([p1[0], obs[0]], [p1[1], obs[1]], '-', color=mypink, alpha=1, 
        linewidth=lw, label=r"$p(o_{\tau} \mid \tilde{x}_{\tau}, m_1)$")

# Likelihood for model 2: distance between p2 prediction and observation
ax.plot([p2[0], obs[0]], [p2[1], obs[1]], '-', color=mygreen, alpha=1, 
        linewidth=lw, label=r"$p(o_{\tau} \mid \tilde{x}_{\tau}, m_2)$")

# Add perpendicular tick marks to likelihood lines (both directions)
draw_orthogonal_line(ax, p1, obs, mypink, lw)
draw_orthogonal_line(ax, p2, obs, mygreen, lw)
draw_orthogonal_line(ax, obs, p1, mypink, lw)
draw_orthogonal_line(ax, obs, p2, mygreen, lw)

# === Configure plot appearance ===
ax.set_xticks([])  # Remove x-axis tick marks
ax.set_yticks([])  # Remove y-axis tick marks
ax.set_xlabel(r'$x_1$', fontsize=FS)  # State dimension 1
ax.set_ylabel(r'$x_2$', fontsize=FS)  # State dimension 2

ax.set_aspect('equal')  # Equal aspect ratio for circular ellipses

# Position legend to the right of the plot
plt.legend(fontsize=FS, loc='center', bbox_to_anchor=[1.23, 0.5])
plt.tight_layout()

# Save figure as PDF
plt.savefig('Fig2_rev.pdf', bbox_inches="tight")
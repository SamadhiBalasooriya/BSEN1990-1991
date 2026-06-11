import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

np.random.seed(7)

# Geometry and timing
L = 10.0   # walkway length (m)
W = 4.0    # walkway width (m)
T = 15.0   # total time (s)
dt = 0.1
times = np.arange(0.0, T + dt, dt)
n_steps = len(times)

# Pedestrians
n_each_side = 10
N = 2 * n_each_side
ped_radius = 0.12

# Obstacle: 2 m long in x-direction, 1 m wide in y-direction
obs_len = 2.0
obs_wid = 1.0
obs_speed = 2.0
obs_y = W / 2.0 - obs_wid / 2.0

# Obstacle motion assumption:
# it moves along the corridor axis (x-direction) and wraps around.
def obstacle_x(t):
    start = -obs_len
    travel = (obs_speed * t) % (L + obs_len + 2.0)
    return start + travel

# Initial pedestrian positions
x_left = np.random.uniform(0.2, 1.5, n_each_side)
y_left = np.linspace(0.4, W - 0.4, n_each_side) + np.random.uniform(-0.12, 0.12, n_each_side)

x_right = np.random.uniform(L - 1.5, L - 0.2, n_each_side)
y_right = np.linspace(0.4, W - 0.4, n_each_side) + np.random.uniform(-0.12, 0.12, n_each_side)

pos = np.vstack([
    np.column_stack([x_left, y_left]),
    np.column_stack([x_right, y_right]),
])

# Desired directions and speeds
direction = np.vstack([
    np.tile(np.array([1.0, 0.0]), (n_each_side, 1)),
    np.tile(np.array([-1.0, 0.0]), (n_each_side, 1)),
])
desired_speed = np.concatenate([
    np.random.normal(1.35, 0.08, n_each_side),
    np.random.normal(1.35, 0.08, n_each_side),
])
vel = direction * desired_speed[:, None] * 0.7

# Storage for animation
history = np.zeros((n_steps, N, 2))
obs_history = np.zeros((n_steps, 2))
history[0] = pos.copy()
obs_history[0] = [obstacle_x(0.0), obs_y]

# Force model parameters
tau = 0.6
wall_gain = 4.0
ped_gain = 0.06
obs_gain = 0.22
max_speed = 1.8

def nearest_point_on_rect(p, rx, ry, rw, rh):
    qx = np.clip(p[0], rx, rx + rw)
    qy = np.clip(p[1], ry, ry + rh)
    return np.array([qx, qy])

# Simulation
for k, t in enumerate(times[1:], start=1):
    rx = obstacle_x(t)
    ry = obs_y

    acc = np.zeros_like(pos)

    # Desired motion
    desired_vel = direction * desired_speed[:, None]
    acc += (desired_vel - vel) / tau

    # Wall repulsion (top and bottom)
    d_bottom = np.maximum(pos[:, 1] - 0.0, 1e-3)
    d_top = np.maximum(W - pos[:, 1], 1e-3)
    acc[:, 1] += wall_gain / (d_bottom ** 2)
    acc[:, 1] -= wall_gain / (d_top ** 2)

    # Pedestrian-pedestrian repulsion
    for i in range(N):
        diff = pos[i] - pos
        dist = np.linalg.norm(diff, axis=1)
        mask = (dist > 1e-6) & (dist < 0.9)
        if np.any(mask):
            rep = (diff[mask] / dist[mask][:, None]) / (dist[mask][:, None] ** 2 + 1e-6)
            acc[i] += ped_gain * rep.sum(axis=0)

    # Obstacle repulsion
    for i in range(N):
        nearest = nearest_point_on_rect(pos[i], rx, ry, obs_len, obs_wid)
        diff = pos[i] - nearest
        dist = np.linalg.norm(diff)

        if rx < L + 0.5 and rx + obs_len > -0.5:
            if dist < 1.0:
                if dist < 1e-6:
                    center = np.array([rx + obs_len / 2.0, ry + obs_wid / 2.0])
                    diff = pos[i] - center
                    dist = np.linalg.norm(diff) + 1e-6
                acc[i] += obs_gain * diff / (dist ** 3 + 1e-6)

                ahead = (direction[i, 0] > 0 and pos[i, 0] < rx + obs_len) or (direction[i, 0] < 0 and pos[i, 0] > rx)
                if ahead and abs(pos[i, 1] - (ry + obs_wid / 2.0)) < 0.8 and abs(pos[i, 0] - (rx + obs_len / 2.0)) < 1.6:
                    side = np.sign(pos[i, 1] - (ry + obs_wid / 2.0))
                    if side == 0:
                        side = np.random.choice([-1.0, 1.0])
                    acc[i, 1] += 0.8 * side

    # Integrate
    vel += acc * dt

    speed = np.linalg.norm(vel, axis=1)
    too_fast = speed > max_speed
    vel[too_fast] *= (max_speed / speed[too_fast])[:, None]

    pos += vel * dt

    # Keep within corridor
    pos[:, 1] = np.clip(pos[:, 1], ped_radius, W - ped_radius)

    # Stop peds from sitting inside obstacle: project out vertically if needed
    inside_x = (pos[:, 0] > rx) & (pos[:, 0] < rx + obs_len)
    inside_y = (pos[:, 1] > ry) & (pos[:, 1] < ry + obs_wid)
    inside = inside_x & inside_y
    if np.any(inside):
        center_y = ry + obs_wid / 2.0
        push_up = pos[inside, 1] >= center_y
        pos_inside = pos[inside].copy()
        pos_inside[push_up, 1] = ry + obs_wid + ped_radius
        pos_inside[~push_up, 1] = ry - ped_radius
        pos[inside] = pos_inside
        vel[inside, 1] *= 0.5

    # Wrap pedestrians so the flow is continuous
    left_to_right = np.arange(n_each_side)
    right_to_left = np.arange(n_each_side, N)

    reset_ltr = pos[left_to_right, 0] > L + 0.3
    if np.any(reset_ltr):
        pos[left_to_right[reset_ltr], 0] = -0.3
        pos[left_to_right[reset_ltr], 1] = np.random.uniform(0.4, W - 0.4, reset_ltr.sum())

    reset_rtl = pos[right_to_left, 0] < -0.3
    if np.any(reset_rtl):
        pos[right_to_left[reset_rtl], 0] = L + 0.3
        pos[right_to_left[reset_rtl], 1] = np.random.uniform(0.4, W - 0.4, reset_rtl.sum())

    history[k] = pos.copy()
    obs_history[k] = [rx, ry]

# Animation
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.set_xlim(0, L)
ax.set_ylim(0, W)
ax.set_aspect('equal')
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
#ax.set_title("20 pedestrians in a 10 m × 4 m walkway with a moving obstacle")

boundary = Rectangle((0, 0), L, W, fill=False)
ax.add_patch(boundary)

scat1 = ax.scatter([], [])
scat2 = ax.scatter([], [])
obs_patch = Rectangle((0, obs_y), obs_len, obs_wid, alpha=0.5)
ax.add_patch(obs_patch)

time_text = ax.text(0.02, 1.03, "", transform=ax.transAxes)

def init():
    scat1.set_offsets(np.empty((0, 2)))
    scat2.set_offsets(np.empty((0, 2)))
    obs_patch.set_xy((obs_history[0, 0], obs_history[0, 1]))
    time_text.set_text("")
    return scat1, scat2, obs_patch, time_text

def update(frame):
    pts1 = history[frame, :n_each_side, :]
    pts2 = history[frame, n_each_side:, :]
    scat1.set_offsets(pts1)
    scat2.set_offsets(pts2)
    obs_patch.set_xy((obs_history[frame, 0], obs_history[frame, 1]))
    time_text.set_text(f"t = {times[frame]:.1f} s")
    return scat1, scat2, obs_patch, time_text

anim = FuncAnimation(
    fig, update, frames=n_steps, init_func=init, interval=100, blit=True
)

anim.save("walkway_moving_obstacle.gif", writer=PillowWriter(fps=10))
plt.show()
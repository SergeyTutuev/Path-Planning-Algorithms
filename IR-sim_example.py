import irsim
from irsim.lib.path_planners import InformedRRTStar
import numpy as np
from shapely.geometry import Point
import random

from plot_overplay import CirclesOverlay
from matplotlib.patches import Circle

env = irsim.make("waypoint_world.yaml")

start = env.get_robot_state()[:2, 0].tolist()   # [x, y]
goal = [47, 47]
safety_margin = 0.4
N_DYNAMIC = 100
WORLD_W, WORLD_H = 50, 50
rate = 2

env_map = env.get_map(resolution=0.05)
planning_radius = env.robot.radius + safety_margin

original_geometry = env.robot.gf._original_geometry
env.robot.gf._original_geometry = Point(0, 0).buffer(planning_radius)

artists_before = set(env._env_plot.ax.get_children())

planner = InformedRRTStar(env_map,  env.robot, max_iter = 10000)
trajectory = planner.planning(start, goal, show_animation=True)

#print(type(trajectory))
#rate = len(trajectory.T) - 1
trajectory = trajectory.T[::rate]
trajectory = trajectory.T

env.robot.gf._original_geometry = original_geometry

if trajectory is None:
    print("RRT не смог найти путь. Увеличьте max_iter или уменьшите expand_dis.")
    env.end()
    exit()


artists_after = set(env._env_plot.ax.get_children())
for art in artists_after - artists_before:
    try:
        art.remove()
    except (ValueError, NotImplementedError, AttributeError):
        pass
env._env_plot.ax.figure.canvas.draw_idle()

env.draw_trajectory(trajectory, traj_type='r-', linewidth=2)
env._env_plot.ax.scatter(goal[0], goal[1], s=120, c='red', zorder=10)

_goal_r = float(env.robot.goal_threshold)
_goal_circle = Circle(
    (0.0, 0.0), radius=_goal_r,
    fill=False, edgecolor="limegreen", linewidth=2.0,
    zorder=9,
)
env._env_plot.ax.add_patch(_goal_circle)

_goal_dot, = env._env_plot.ax.plot(
    [], [], marker="+", linestyle="none",
    color="limegreen", markersize=16, markeredgewidth=2.0, zorder=10,
)

def set_goal_visual(gx: float, gy: float) -> None:
    """Перенести круг и маркер цели на (gx, gy)."""
    _goal_circle.set_center((gx, gy))
    _goal_dot.set_data([gx], [gy])

def make_convex_polygon_shape():
    avg_r = random.uniform(0.2, 0.5)
    return {
        "name": "polygon",
        "random_shape": True,
        "avg_radius_range": [avg_r, avg_r],   # фиксируем выбранный радиус
        "num_vertices_range": [4, 6],
        "irregularity_range": [0.1, 0.4],
        "spikeyness_range": [0.0, 0.1],
        "is_convex": True }

dynamic_obstacles = []
for i in range(N_DYNAMIC):
    start_x = random.uniform(5, WORLD_W - 5)
    start_y = random.uniform(5, WORLD_H - 5)

    obs = env.object_factory.create_obstacle(
        kinematics={"name": "omni"},
        shape=make_convex_polygon_shape(),
        state=[start_x, start_y, 0.0],
        vel_max=[0.5, 1.0],
        #goal=[goal_x, goal_y, 0.0],
        behavior={"name": "dash",  "wander": True,  "range_low": [1, 1, -3.14],  "range_high": [50, 50, 3.14], "acce": 0.2},  #"factor": 0.05,  }, "target_roles": "obstacle"},
        plot={"show_goal": False, "show_trajectory": False,  "show_trail": False} )
    dynamic_obstacles.append(obs)

env.add_objects(dynamic_obstacles)
print(f"Добавлено {len(dynamic_obstacles)} подвижных препятствий.")

waypoints = trajectory.T.tolist()
waypoints = waypoints[::-1]
print(f"Построено точек пути: {len(waypoints)}")
current_idx = 0
env.robot.set_goal([waypoints[current_idx][0], waypoints[current_idx][1], 0])
set_goal_visual(waypoints[current_idx][0], waypoints[current_idx][1])

ax = env._env_plot.ax                         # тот же axis, что ir-sim использует для рендера
overlay = CirclesOverlay(ax, color="tab:orange", alpha=0.45, linewidth=0.7, zorder=8)


def collect_circles(env):
    """Собрать debug-круги со всех объектов (робот + препятствия с RVO)."""
    out = []
    for obj in env.objects:
        c = getattr(obj, "_debug_circles", None)
        if c:
            out.extend(c)
    return out


for i, _ in enumerate(range(5000)):
    env.step()
    overlay.update(collect_circles(env))
    env.render(0.01)

        
    robot_xy = env.robot.state[:2, 0]
    target_xy = np.array(waypoints[current_idx][:2])
    dist = np.linalg.norm(robot_xy - target_xy)

    if dist < env.robot.goal_threshold:
        current_idx += 1
        if current_idx >= len(waypoints):
            print("GOAL достигнута!")
            break
        print(waypoints[current_idx][0], waypoints[current_idx][1])
        env.robot.set_goal([waypoints[current_idx][0], waypoints[current_idx][1], 0])
        set_goal_visual(waypoints[current_idx][0], waypoints[current_idx][1])
        env.robot.arrive_flag = False

env.end()
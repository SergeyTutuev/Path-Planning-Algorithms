from typing import Any

import numpy as np

from irsim.lib import reciprocal_vel_obs, register_behavior, social_force_model

from math import atan, atan2, cos, pi, sin

from irsim.util.util import (
    WrapToPi,
    log_warning_once,      # NEW
    relative_position,
    vel_world2diff,
    vel_world2omni,
)

"""
Behavior Methods Module

This module contains behavior functions for different types of robots in the IR-Sim environment.
It provides implementations for:

1. Registered behavior functions (decorated with @register_behavior):
   - beh_diff_rvo: Differential drive robot with RVO behavior
   - beh_diff_dash: Differential drive robot with dash-to-goal behavior
   - beh_diff_sfm: Differential drive robot with Social Force Model behavior
   - beh_omni_dash: Omnidirectional robot with dash-to-goal behavior
   - beh_omni_rvo: Omnidirectional robot with RVO behavior
   - beh_omni_sfm: Omnidirectional robot with Social Force Model behavior
   - beh_acker_dash: Ackermann steering robot with dash-to-goal behavior

2. Core behavior calculation functions:
   - OmniRVO: Omnidirectional RVO velocity calculation
   - DiffRVO: Differential drive RVO velocity calculation
   - OmniDash: Omnidirectional dash-to-goal velocity calculation
   - DiffDash: Differential drive dash-to-goal velocity calculation
   - AckerDash: Ackermann steering dash-to-goal velocity calculation
   - SFMVelocity: Social Force Model velocity calculation

These functions are designed to be used with the robot behavior system to control
robot movements in various scenarios including collision avoidance (RVO) and
goal-reaching behaviors (dash).

See :mod:`irsim.lib.algorithm.social_force_model` for the SFM algorithmic
details.
"""


def _goal_pending(ego_object: Any, behavior: str) -> bool:
    """
    Report whether a behavior is still waiting for a goal to be configured.

    Args:
        ego_object: The ego robot object.
        behavior (str): Behavior name, used in the periodic warning.

    Returns:
        bool: True while ``ego_object`` has no goal, in which case the caller
        should return a zero velocity of its own shape.
    """

    if ego_object.goal is not None:
        return False

    ego_object.logger.warning_once(
        f"{ego_object.name}: goal is None, the {behavior} behavior is waiting for a goal"
    )

    return True


def _rvo_params(kwargs):
    return (
        kwargs.get("vxmax", 1.5),
        kwargs.get("vymax", 1.5),
        kwargs.get("acce", 1.0),
        kwargs.get("factor", 1.0),
        kwargs.get("mode", "rvo"),
        kwargs.get("neighbor_threshold", 3.0),
        kwargs.get("decel", None),
        kwargs.get("allow_backward", True),
        kwargs.get("backward_penalty", 0.0),
        kwargs.get("save_marg", 0.0),
        kwargs.get("rad_approx", None),
        kwargs.get("rad_approx_overlap", 0.5),
        kwargs.get("rad_approx_range", 10.0),
        kwargs.get("k_heading", 2.0),
        kwargs.get("dtheta_deadband", 0.02),
        kwargs.get("idle_penalty", 0.0),
        kwargs.get("idle_grace", 3.0),
        kwargs.get("idle_penalty_cap", 10.0),
        kwargs.get("idle_speed_thresh", 0.1),
    )

def _track_idle_time(ego_object, idle_speed_thresh: float) -> float:
    """Сколько секунд робот уже «практически стоит» (|v_linear| < порога).
    Счётчик живёт на ego_object — переживает пересоздание reciprocal_vel_obs."""
    dt = float(ego_object._world_param.step_time)
    st = np.asarray(ego_object.state).ravel()
    speed = float(np.hypot(st[2], st[3])) if st.size >= 4 else 0.0
    if speed < float(idle_speed_thresh):
        ego_object._rvo_idle_time = getattr(ego_object, "_rvo_idle_time", 0.0) + dt
    else:
        ego_object._rvo_idle_time = 0.0
    return float(ego_object._rvo_idle_time)

# ---------------------------------------------------------------------------
# Извлечение рёбер полигонов (ir-sim не заполняет rvo_line_segments для polygon)
# ---------------------------------------------------------------------------
try:
    from shapely import affinity as _shapely_affinity
except ImportError:                       # shapely не установлен — работаем без этого
    _shapely_affinity = None


def _world_geometry(obj):
    """Геометрия объекта в мировой СК. Сначала штатный атрибут, иначе — трансформ."""
    gf = getattr(obj, "gf", None)
    if gf is None:
        return None

    # 1) если ir-sim отдаёт мировую геометрию — используем её
    geom = getattr(gf, "geometry", None)
    if geom is not None:
        return geom

    # 2) иначе берём локальную и трансформируем по state
    base = getattr(gf, "_original_geometry", None)
    if base is None:
        return None
    if _shapely_affinity is None:
        return base

    st = np.asarray(obj.state).ravel()
    if st.size < 2:
        return base
    x, y = float(st[0]), float(st[1])
    th = float(st[2]) if st.size >= 3 else 0.0
    c, s = float(np.cos(th)), float(np.sin(th))
    return _shapely_affinity.affine_transform(base, [c, -s, s, c, x, y])


def _extract_edges(geom):
    """Рёбра shapely Polygon / MultiPolygon / LineString в формате [[x1,y1,x2,y2], ...]."""
    if geom is None:
        return []
    if hasattr(geom, "exterior"):                     # Polygon
        coords = list(geom.exterior.coords)
        return [[coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]]
                for i in range(len(coords) - 1)]
    if hasattr(geom, "geoms"):                        # MultiPolygon / MultiLineString
        out = []
        for g in geom.geoms:
            out.extend(_extract_edges(g))
        return out
    if hasattr(geom, "coords"):                       # LineString
        coords = list(geom.coords)
        return [[coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]]
                for i in range(len(coords) - 1)]
    return []

def vel_world2acker(state, vel_world, wheelbase, v_max, delta_max, dt,
                    k_heading=2.0, dtheta_deadband=0.02,
                    allow_backward=False):          # NEW
    vx = float(vel_world[0]); vy = float(vel_world[1])
    th = float(state[2, 0]) if state.shape[0] > 2 else 0.0

    v_fwd = vx * cos(th) + vy * sin(th)

    # NEW: не разрешаем задний ход (согласовано с mask)
    if not allow_backward and v_fwd < 0.0:
        v_fwd = 0.0
    v_fwd = float(np.clip(v_fwd, -v_max, v_max))

    if abs(vx) + abs(vy) < 1e-6:
        return np.array([[0.0], [0.0]])

    phi = atan2(vy, vx)
    dtheta = (phi - th + pi) % (2.0 * pi) - pi

    # NEW: чисто боковая скорость (v_fwd ≈ 0) — acker так не умеет.
    # Крутимся, ползя вперёд: U-turn, но без заноса зада в стену.
    if abs(v_fwd) < 1e-3:
        if abs(dtheta) < dtheta_deadband:
            return np.array([[0.0], [0.0]])
        v_creep = min(0.3 * v_max, 0.25)
        delta = np.sign(dtheta) * delta_max
        return np.array([[v_creep], [delta]])

    if abs(dtheta) < dtheta_deadband:
        return np.array([[v_fwd], [0.0]])

    w_des = k_heading * dtheta
    L = max(float(wheelbase), 1e-3)
    delta = atan(L * w_des / v_fwd) if abs(v_fwd) > 1e-3 else 0.0
    delta = float(np.clip(delta, -delta_max, delta_max))
    return np.array([[v_fwd], [delta]])

def _polygon_segments(obj, min_radius=0.8):
    
    """Рёбра крупного полигона в мировой СК.

    * ``min_radius`` — порог по ``rvo_neighbor_state[4]`` (описанный радиус).
      Мелкие объекты (динамические с avg_radius 0.2–0.5 → описанный 0.3–0.7)
      оставляем как круги — это корректно и быстрее.
    * Результат кэшируется в ``obj.rvo_line_segments``: для статичных
      полигонов извлечение выполняется **один раз**, а не каждый шаг.
    """
    segs = getattr(obj, "rvo_line_segments", None)
    if segs:
        return segs

    ns = getattr(obj, "rvo_neighbor_state", None)
    if not ns or len(ns) < 5 or float(ns[4]) < min_radius:
        return []

    geom = _world_geometry(obj)
    segs = _extract_edges(geom)
    if segs:
        try:
            obj.rvo_line_segments = segs      # кэш (для статичных — навсегда)
        except Exception:
            pass
    return segs

@register_behavior("diff", "rvo")
def beh_diff_rvo(ego_object, external_objects, **kwargs):


    if _goal_pending(ego_object, "rvo"):
        return np.zeros((2, 1))

    rvo_neighbor = []
    line_segments = []
    for obj in external_objects:
        segs = _polygon_segments(obj)
        if segs:
            line_segments.extend(segs)
        else:
            rvo_neighbor.append(obj.rvo_neighbor_state)

    rvo_state = ego_object.rvo_state
    (vxmax, vymax, acce, factor, mode, neighbor_threshold,
 decel, allow_backward, backward_penalty, save_marg,
 rad_approx, rad_approx_overlap, rad_approx_range,_, _,
 idle_penalty, idle_grace, idle_penalty_cap, idle_speed_thresh) = _rvo_params(kwargs)
    idle_time = _track_idle_time(ego_object, idle_speed_thresh)
    ego_object._debug_circles = getattr(ego_object, "_debug_circles", [])

    return DiffRVO(
        rvo_state, rvo_neighbor, vxmax, vymax, acce, factor, 
        decel, allow_backward, backward_penalty, save_marg, mode,
        neighbor_threshold,
        line_segments=line_segments,
        rad_approx=rad_approx,
        rad_approx_overlap=rad_approx_overlap,
        rad_approx_range=rad_approx_range,
        circles_store=ego_object._debug_circles,
        idle_penalty = idle_penalty, idle_grace = idle_grace,
        idle_penalty_cap = idle_penalty_cap, idle_time = idle_time
    )

@register_behavior("acker", "rvo")
def beh_acker_rvo(ego_object, external_objects, **kwargs):
    
    if _goal_pending(ego_object, "rvo"):
        return np.zeros((2, 1))

    # wheelbase: kwargs → shape → default L=1.0
    wheelbase = kwargs.get("wheelbase", None)
    if wheelbase is None:
        shape = getattr(ego_object, "shape", None)
        wheelbase = getattr(shape, "wheelbase", None) if shape is not None else None
    if wheelbase is None:
        log_warning_once(
            f"{ego_object.name}: beh_acker_rvo wheelbase is None; "
            "using default L=1.0 (diff-like behavior). "
            "Set `wheelbase` in YAML for accurate Ackermann."
        )
        wheelbase = 1.0
    wheelbase = float(wheelbase)

    # лимиты из vel_max[linear, steering_angle]
    vmax_arr = np.asarray(ego_object.vel_max, dtype=float)
    max_v = float(vmax_arr[0, 0])
    delta_max = float(kwargs.get("delta_max", vmax_arr[1, 0]))
    dt = float(ego_object._world_param.step_time)

    # внешние объекты → (круги, отрезки)
    rvo_neighbor, line_segments = [], []
    for obj in external_objects:
        segs = _polygon_segments(obj)
        if segs:
            line_segments.extend(segs)
        else:
            rvo_neighbor.append(obj.rvo_neighbor_state)

    (vxmax, vymax, acce, factor, mode, neighbor_threshold,
     decel, allow_backward, backward_penalty, save_marg,
     rad_approx, rad_approx_overlap, rad_approx_range, k_heading, dtheta_deadband,
     idle_penalty, idle_grace, idle_penalty_cap, idle_speed_thresh) = _rvo_params(kwargs)

    idle_time = _track_idle_time(ego_object, idle_speed_thresh)

    vxmax = kwargs.get("vxmax", float(vmax_arr[0, 0]))
    vymax = kwargs.get("vymax", float(vmax_arr[0, 0]))
    ego_object._debug_circles = getattr(ego_object, "_debug_circles", [])

    return AckerRVO(
        ego_object.rvo_state,
        rvo_neighbor,
        vxmax, vymax, acce, factor, 
        decel, allow_backward, backward_penalty, save_marg,
        mode=mode,
        neighbor_threshold=neighbor_threshold,
        line_segments=line_segments,
        rad_approx=rad_approx,
        rad_approx_overlap=rad_approx_overlap,
        rad_approx_range=rad_approx_range,
        circles_store=ego_object._debug_circles,
        wheelbase=wheelbase,
        delta_max=delta_max,
        dt=dt,
        max_v=max_v,
        ego_theta=float(ego_object.state[2, 0]), k_heading = k_heading, dtheta_deadband= dtheta_deadband,
        idle_penalty = idle_penalty, idle_grace = idle_grace,
        idle_penalty_cap = idle_penalty_cap, idle_time = idle_time
    )


@register_behavior("diff", "dash")
def beh_diff_dash(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:
    """
    Behavior function for differential drive robot using dash-to-goal behavior.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: Additional keyword arguments:
            - angle_tolerance (float): Allowable angular deviation, default 0.1.

    Returns:
        np.array: Velocity [linear, angular] (2x1) for differential drive.
    """

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    max_vel = ego_object.vel_max
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    angular_acce = ego_object.info.acce[1, 0]
    dt = ego_object._world_param.step_time

    if _goal_pending(ego_object, "dash"):
        return np.zeros((2, 1))

    vel = DiffDash(
        state, goal, max_vel, goal_threshold, angle_tolerance, angular_acce, dt
    )
    min_step, max_step = ego_object.get_vel_range()
    return np.clip(vel, min_step, max_step)


@register_behavior("omni", "dash")
def beh_omni_dash(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:
    """
    Behavior function for omnidirectional robot using dash-to-goal behavior.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        np.array: Velocity [vx, vy] (2x1) for omnidirectional drive.
    """

    if _goal_pending(ego_object, "dash"):
        return np.zeros((2, 1))

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    max_vel = ego_object.vel_max
    vel = OmniDash(state, goal, max_vel, goal_threshold)
    min_step, max_step = ego_object.get_vel_range()
    return np.clip(vel, min_step, max_step)


@register_behavior("omni", "rvo")
def beh_omni_rvo(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:
    """
    Behavior function for omnidirectional robot using RVO (Reciprocal Velocity Obstacles).

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: Additional keyword arguments:
            - vxmax (float): Maximum x velocity, default 1.5.
            - vymax (float): Maximum y velocity, default 1.5.
            - acce (float): Acceleration factor, default 1.0.
            - factor (float): Additional scaling factor, default 1.0.
            - mode (str): RVO calculation mode, default "rvo".
            - neighbor_threshold (float): Neighbor threshold distance, default 3.0.

    Returns:
        np.array: Velocity [vx, vy] (2x1) for omnidirectional drive.
    """

    if _goal_pending(ego_object, "rvo"):
        return np.zeros((2, 1))

    rvo_neighbor = []
    line_segments = []
    for obj in external_objects:
        segs = obj.rvo_line_segments
        if segs:
            line_segments.extend(segs)
        else:
            rvo_neighbor.append(obj.rvo_neighbor_state)
    rvo_state = ego_object.rvo_state
    vxmax, vymax, acce, factor, mode, neighbor_threshold = _rvo_params(kwargs)
    world_vel = OmniRVO(
        rvo_state,
        rvo_neighbor,
        vxmax,
        vymax,
        acce,
        factor,
        mode,
        neighbor_threshold,
        line_segments=line_segments,
    )
    return vel_world2omni(ego_object.state[2, 0], world_vel)


@register_behavior("diff", "sfm")
def beh_diff_sfm(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:
    """
    Behavior function for differential drive robot using the Social Force Model.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: SFM tuning parameters; see :func:`SFMVelocity`.

    Returns:
        np.array: Velocity [linear, angular] (2x1) for differential drive.
    """
    if _goal_pending(ego_object, "sfm"):
        return np.zeros((2, 1))

    neighbors = []
    line_segments = []
    for obj in external_objects:
        segs = obj.rvo_line_segments
        if segs:
            line_segments.extend(segs)
        else:
            neighbors.append(obj.rvo_neighbor_state)

    vx, vy = SFMVelocity(
        ego_object.rvo_state,
        neighbors,
        line_segments,
        step_time=ego_object._world_param.step_time,
        **kwargs,
    )
    # SFM produces a holonomic ``(vx, vy)``; track it tightly so the small
    # sideways component from anisotropic social/obstacle forces doesn't
    # get clipped by ``vel_world2diff``'s default deadband.
    _, vmax_pair = ego_object.get_vel_range()
    w_max = float(vmax_pair[1, 0])
    return vel_world2diff(
        ego_object.rvo_state[-1],
        [vx, vy],
        w_max=w_max,
        guarantee_time=ego_object._world_param.step_time,
        tolerance=0.02,
    )


@register_behavior("omni", "sfm")
def beh_omni_sfm(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:
    """
    Behavior function for omnidirectional robot using the Social Force Model.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: SFM tuning parameters; see :func:`SFMVelocity`.

    Returns:
        np.array: Velocity [vx, vy] (2x1) for omnidirectional drive.
    """
    if _goal_pending(ego_object, "sfm"):
        return np.zeros((2, 1))

    neighbors = []
    line_segments = []
    for obj in external_objects:
        segs = obj.rvo_line_segments
        if segs:
            line_segments.extend(segs)
        else:
            neighbors.append(obj.rvo_neighbor_state)

    vx, vy = SFMVelocity(
        ego_object.rvo_state,
        neighbors,
        line_segments,
        step_time=ego_object._world_param.step_time,
        **kwargs,
    )
    return vel_world2omni(ego_object.state[2, 0], np.array([[vx], [vy]]))


@register_behavior("omni_angular", "dash")
def beh_omni_angular_dash(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:
    """
    Behavior function for omnidirectional-angular robot using dash-to-goal behavior.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: Additional keyword arguments:
            - angle_tolerance (float): Allowable angular deviation, default 0.1.

    Returns:
        np.array: Velocity [forward, lateral, yaw_rate] (3x1) in body frame.
    """

    if _goal_pending(ego_object, "dash"):
        return np.zeros((3, 1))

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    max_vel = ego_object.vel_max
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    acce = ego_object.info.acce
    angular_acce = acce[2, 0] if acce.shape[0] > 2 else float("inf")
    dt = ego_object._world_param.step_time
    vel = OmniAngularDash(
        state, goal, max_vel, goal_threshold, angle_tolerance, angular_acce, dt
    )
    min_step, max_step = ego_object.get_vel_range()
    return np.clip(vel, min_step, max_step)


@register_behavior("acker", "dash")
def beh_acker_dash(
    ego_object: Any, external_objects: list[Any], **kwargs: Any
) -> np.ndarray:
    """
    Behavior function for Ackermann steering robot using dash-to-goal behavior.

    Args:
        ego_object: The ego robot object.
        external_objects (list): List of external objects in the environment.
        **kwargs: Additional keyword arguments:
            - angle_tolerance (float): Allowable angular deviation, default 0.1.

    Returns:
        np.array: Velocity [linear, steering angle] (2x1) for Ackermann drive.
    """

    if _goal_pending(ego_object, "dash"):
        return np.zeros((2, 1))

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    max_vel = ego_object.vel_max
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    vel = AckerDash(state, goal, max_vel, goal_threshold, angle_tolerance)
    min_step, max_step = ego_object.get_vel_range()
    return np.clip(vel, min_step, max_step)


def SFMVelocity(
    state_tuple: Any,
    neighbor_list: list[Any] | None = None,
    line_segments: list[list[float]] | None = None,
    vmax: float = 1.5,
    step_time: float = 0.1,
    neighbor_threshold: float = 10.0,
    relaxation_time: float = 0.5,
    force_factor_desired: float = 1.0,
    force_factor_social: float = 2.1,
    force_factor_obstacle: float = 10.0,
    sigma_obstacle: float = 0.8,
    lambda_importance: float = 2.0,
    gamma: float = 0.35,
    n_angular: float = 2.0,
    n_velocity: float = 3.0,
    safety_radius: float = 0.0,
    **_: Any,
) -> tuple[float, float]:
    """
    Compute the next world-frame velocity using the Social Force Model.

    Anisotropic Moussaid-Helbing (2009) variant.

    Args:
        state_tuple: Full RVO-style state
            ``[x, y, vx, vy, radius, vx_des, vy_des, theta]``.
        neighbor_list: Neighbour states ``[[x, y, vx, vy, radius], ...]``.
        line_segments: Line obstacles ``[[x1, y1, x2, y2], ...]``.
        vmax: Speed cap.
        step_time: Integration step.
        neighbor_threshold: Spatial cutoff for social interactions.
        relaxation_time, force_factor_*, sigma_obstacle, lambda_importance, gamma, n_angular, n_velocity:
            SFM tuning parameters (see :class:`social_force_model`).

    Returns:
        tuple[float, float]: Updated world-frame velocity ``(vx, vy)``.
    """
    if neighbor_list is None:
        neighbor_list = []
    if line_segments is None:
        line_segments = []

    x, y = state_tuple[0], state_tuple[1]
    filtered_neighbors = [
        nb
        for nb in neighbor_list
        if (x - nb[0]) ** 2 + (y - nb[1]) ** 2 < neighbor_threshold**2
    ]

    # ``rvo_state`` provides ``(vx_des, vy_des)`` with components scaled by
    # the kinematics-specific ``vel_max`` (anisotropic for diff/acker).
    # SFM expects ``desired_direction * vmax``, so renormalise here.
    state_tuple = list(state_tuple)
    vx_des, vy_des = state_tuple[5], state_tuple[6]
    norm = (vx_des * vx_des + vy_des * vy_des) ** 0.5
    if norm > 1e-9:
        state_tuple[5] = vmax * vx_des / norm
        state_tuple[6] = vmax * vy_des / norm

    sfm = social_force_model(
        state=state_tuple,
        neighbor_list=filtered_neighbors,
        line_obs_list=line_segments,
        vmax=vmax,
        step_time=step_time,
        relaxation_time=relaxation_time,
        force_factor_desired=force_factor_desired,
        force_factor_social=force_factor_social,
        force_factor_obstacle=force_factor_obstacle,
        sigma_obstacle=sigma_obstacle,
        lambda_importance=lambda_importance,
        gamma=gamma,
        n_angular=n_angular,
        n_velocity=n_velocity,
        neighbor_range=neighbor_threshold,
        safety_radius=safety_radius,
    )
    vx, vy = sfm.cal_vel()
    return vx, vy


def OmniRVO(
    state_tuple: Any,
    neighbor_list: list[Any] | None = None,
    vxmax: float = 1.5,
    vymax: float = 1.5,
    acce: float = 1,
    factor: float = 1.0,
    mode: str = "rvo",
    neighbor_threshold: float = 3.0,
    line_segments: list[list[float]] | None = None,
) -> np.ndarray:
    """
    Calculate the omnidirectional velocity using RVO.

    Args:
        state_tuple (tuple): Current state and orientation.
        neighbor_list (list): List of neighboring agents (default None).
        vxmax (float): Maximum x velocity (default 1.5).
        vymax (float): Maximum y velocity (default 1.5).
        acce (float): Acceleration factor (default 1).
        factor (float): Additional scaling factor (default 1.0).
        mode (str): RVO calculation mode (default "rvo").
        neighbor_threshold (float): Neighbor threshold (default 3.0).
        line_segments (list): Line segments [[x1, y1, x2, y2], ...] (default None).

    Returns:
        np.array: Velocity [vx, vy] (2x1).
    """

    if neighbor_list is None:
        neighbor_list = []

    x, y = state_tuple[0], state_tuple[1]

    filtered_neighbor_list = [
        neighbor
        for neighbor in neighbor_list
        if (x - neighbor[0]) ** 2 + (y - neighbor[1]) ** 2 < neighbor_threshold**2
    ]

    rvo_behavior = reciprocal_vel_obs(
        state_tuple,
        filtered_neighbor_list,
        vxmax,
        vymax,
        acce,
        factor,
        line_obs_list=line_segments,
    )
    rvo_vel = rvo_behavior.cal_vel(mode)

    return np.array([[rvo_vel[0]], [rvo_vel[1]]])


def DiffRVO(state_tuple, neighbor_list=None,
            vxmax=1.5, vymax=1.5, acce=1, factor=1.0, 
            decel=None, allow_backward=True, backward_penalty=0.0, save_marg=0.0,
            mode="rvo", neighbor_threshold=3.0, line_segments=None,
            rad_approx=None, rad_approx_overlap=0.5, rad_approx_range=10.0,
            circles_store=None, idle_penalty=0.0, idle_grace=3.0,
            idle_penalty_cap=10.0, idle_time=0.0):
    if neighbor_list is None:
        neighbor_list = []
    if line_segments is None:
        line_segments = []

    x, y = state_tuple[0], state_tuple[1]
    filtered_neighbor_list = [
        nb for nb in neighbor_list
        if (x - nb[0]) ** 2 + (y - nb[1]) ** 2 < neighbor_threshold ** 2
    ]

    rvo_behavior = reciprocal_vel_obs(
        state_tuple, filtered_neighbor_list, vxmax, vymax, acce, factor,
        line_obs_list=line_segments,
        rad_approx=rad_approx,
        rad_approx_overlap=rad_approx_overlap,
        rad_approx_range=rad_approx_range,
        decel=decel,
        allow_backward=allow_backward,
        backward_penalty=backward_penalty,
        save_marg=save_marg,
        idle_penalty=idle_penalty,
        idle_grace=idle_grace,
        idle_penalty_cap=idle_penalty_cap,
        idle_time=idle_time
    )

    if circles_store is not None:                       # ← ОДИН блок
        circles_store.clear()
        circles_store.extend(rvo_behavior.get_active_line_circles())
        for c in filtered_neighbor_list:
            circles_store.append((c[0], c[1], c[2], c[3], c[4]))

    rvo_vel = rvo_behavior.cal_vel(mode)                # ← ОДИН вызов
    diff_vel = vel_world2diff(state_tuple[-1], rvo_vel)

    if not diff_vel.any() and not filtered_neighbor_list and not line_segments:
        diff_vel = vel_world2diff(state_tuple[-1], [state_tuple[5], state_tuple[6]])
        diff_vel[0, 0] = 0.0

    return diff_vel

def AckerRVO(
    state_tuple, neighbor_list=None,
    vxmax=1.5, vymax=1.5, acce=1, factor=1.0,
    decel=None, allow_backward=True, backward_penalty=0.0, save_marg=0.0,
    mode = 'rvo',
    neighbor_threshold=3.0, line_segments=None,
    rad_approx=None, rad_approx_overlap=0.5, rad_approx_range=10.0,
    circles_store=None,
    wheelbase=1.0, delta_max=0.5, dt=0.1, max_v=1.5,
    ego_theta=None, k_heading=2.0, dtheta_deadband=0.02,
    idle_penalty: float = 0.0, idle_grace: float = 3.0,
    idle_penalty_cap: float = 10.0, idle_time: float = 0.0
):
    if neighbor_list is None:
        neighbor_list = []
    if line_segments is None:
        line_segments = []

    x, y = state_tuple[0], state_tuple[1]
    filtered_neighbor_list = [
        nb for nb in neighbor_list
        if (x - nb[0]) ** 2 + (y - nb[1]) ** 2 < neighbor_threshold ** 2
    ]

    if ego_theta is None:
        ego_theta = float(state_tuple[7]) if len(state_tuple) >= 8 else 0.0

    rvo_behavior = reciprocal_vel_obs(
    state_tuple, filtered_neighbor_list, vxmax, vymax, acce, factor,
    line_obs_list=line_segments,
    decel=decel, allow_backward=allow_backward,
    backward_penalty=backward_penalty, save_marg=save_marg,
    rad_approx=rad_approx, rad_approx_overlap=rad_approx_overlap,
    rad_approx_range=rad_approx_range,
    idle_penalty=idle_penalty,
    idle_grace=idle_grace,
    idle_penalty_cap=idle_penalty_cap,
    idle_time=idle_time,
    kinematics="acker",
    wheelbase=wheelbase,
    delta_max=delta_max,
    ego_theta=ego_theta,
    )

    if circles_store is not None:
        circles_store.clear()
        circles_store.extend(rvo_behavior.get_active_line_circles())
        for c in filtered_neighbor_list:
            circles_store.append((c[0], c[1], c[2], c[3], c[4]))

    rvo_vel = rvo_behavior.cal_vel(mode)

    state_xy_th = np.array([[state_tuple[0]], [state_tuple[1]], [ego_theta]])
    return vel_world2acker(state_xy_th, rvo_vel, wheelbase, max_v, delta_max, dt, k_heading=k_heading, 
                           dtheta_deadband=dtheta_deadband, allow_backward=allow_backward)

def OmniDash(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.1,
) -> np.ndarray:
    """
    Calculate the body-frame velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1 or 2x1).
        goal (np.array): Goal position [x, y] (2x1).
        max_vel (np.array): Maximum velocity [forward, lateral] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).

    Returns:
        np.array: Body-frame velocity [forward, lateral] (2x1).
    """
    distance, radian = relative_position(state, goal)
    theta = state[2, 0] if state.shape[0] > 2 else 0.0

    if distance > goal_threshold:
        diff_angle = WrapToPi(radian - theta)
        forward = max_vel[0, 0] * cos(diff_angle)
        lateral = max_vel[1, 0] * sin(diff_angle)
    else:
        forward = 0
        lateral = 0

    return np.array([[forward], [lateral]])


def OmniAngularDash(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.3,
    angle_tolerance: float = 0.1,
    angular_acce: float = float("inf"),
    dt: float = 0.0,
) -> np.ndarray:
    """
    Calculate body-frame velocity to reach a goal.

    Drives forward and strafes laterally toward the goal while turning
    to face it. After arriving at the goal position, rotates in place
    to match the goal orientation.

    The yaw rate is ramped down near the target so the robot can decelerate
    to a stop within the remaining angle. The ramp uses the exact discrete-time
    formula ``ω ≤ -a·dt + √(a²·dt² + 2·a·remaining)`` so the robot stops
    cleanly within one step of ``angle_tolerance`` without overshoot.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal state [x, y, theta] (3x1).
        max_vel (np.array): Absolute maximum velocity [forward, lateral, yaw_rate] (3x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.3).
        angle_tolerance (float): Allowable angular deviation (default 0.1).
        angular_acce (float): Angular acceleration limit (rad/s^2). Used to cap
            yaw rate so the robot can always stop within the remaining angle.
            ``inf`` (default) gives bang-bang behaviour.
        dt (float): Simulation time step (s). Used for exact discrete-time decel
            ramp. 0 (default) falls back to the continuous-time approximation.

    Returns:
        np.array: Body-frame velocity [forward, lateral, yaw_rate] (3x1).
    """
    distance, radian = relative_position(state, goal)
    theta = state[2, 0]

    if distance > goal_threshold:
        diff_angle = WrapToPi(radian - theta)
        forward = max_vel[0, 0] * cos(diff_angle)
        lateral = max_vel[1, 0] * sin(diff_angle)
        target_angle = radian
    else:
        forward = 0
        lateral = 0
        target_angle = goal[2, 0] if goal.shape[0] > 2 else theta

    diff_angle = WrapToPi(target_angle - theta)
    if abs(diff_angle) <= angle_tolerance:
        yaw_rate = 0.0
    else:
        if np.isfinite(angular_acce) and angular_acce > 0:
            remaining = abs(diff_angle) - angle_tolerance
            # Exact discrete-time limit: robot moves ω·dt this step before braking,
            # so the true stopping condition is ω·dt + ω²/(2a) ≤ remaining.
            # Solving: ω ≤ -a·dt + √(a²·dt² + 2·a·remaining)
            a_dt = angular_acce * dt
            max_yaw = -a_dt + np.sqrt(a_dt**2 + 2.0 * angular_acce * remaining)
            # The decel ramp grows unbounded with ``remaining``; never exceed
            # the configured yaw-rate limit.
            max_yaw = min(abs(max_vel[2, 0]), max_yaw)
        else:
            max_yaw = abs(max_vel[2, 0])
        yaw_rate = np.sign(diff_angle) * max_yaw

    return np.array([[forward], [lateral], [yaw_rate]])


def DiffDash(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float = 0.1,
    angle_tolerance: float = 0.2,
    angular_acce: float = float("inf"),
    dt: float = 0.0,
) -> np.ndarray:
    """
    Calculate the differential drive velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y] (2x1).
        max_vel (np.array): Absolute maximum velocity [linear, angular] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).
        angle_tolerance (float): Allowable angular deviation (default 0.2).
        angular_acce (float): Angular acceleration limit (rad/s^2) for the decel ramp.
            ``inf`` (default) gives bang-bang behaviour.
        dt (float): Simulation time step (s) for exact discrete-time decel ramp.

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    distance, radian = relative_position(state, goal)

    if distance < goal_threshold:
        return np.zeros((2, 1))

    diff_radian = WrapToPi(radian - state[2, 0])
    linear = max_vel[0, 0] * np.cos(diff_radian)

    if abs(diff_radian) < angle_tolerance:
        angular = 0.0
    else:
        if np.isfinite(angular_acce) and angular_acce > 0:
            remaining = abs(diff_radian) - angle_tolerance
            a_dt = angular_acce * dt
            max_ang = -a_dt + np.sqrt(a_dt**2 + 2.0 * angular_acce * remaining)
            # The decel ramp grows unbounded with ``remaining``; never exceed
            # the configured angular-speed limit.
            max_ang = min(abs(max_vel[1, 0]), max_ang)
        else:
            max_ang = abs(max_vel[1, 0])
        angular = np.sign(diff_radian) * max_ang

    return np.array([[linear], [angular]])


def AckerDash(
    state: np.ndarray,
    goal: np.ndarray,
    max_vel: np.ndarray,
    goal_threshold: float,
    angle_tolerance: float,
) -> np.ndarray:
    """
    Calculate the Ackermann steering velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y] (2x1).
        max_vel (np.array): Maximum velocity [linear, steering angle] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached.
        angle_tolerance (float): Allowable angular deviation.

    Returns:
        np.array: Velocity [linear, steering angle] (2x1).
    """
    dis, radian = relative_position(state, goal)
    steer_opt = 0.0
    diff_radian = WrapToPi(radian - state[2, 0])

    if diff_radian > -angle_tolerance and diff_radian < angle_tolerance:
        diff_radian = 0

    if dis < goal_threshold:
        v_opt, steer_opt = 0, 0
    else:
        v_opt = max_vel[0, 0]
        steer_opt = np.clip(diff_radian, -max_vel[1, 0], max_vel[1, 0])

    return np.array([[v_opt], [steer_opt]])

# --- optional: DWA behavior, registered only if user's dwa.py is importable ---
try:
    from . import dwa_behavior_methods  # noqa: F401
except ImportError as _e:
    from irsim.util.util import log_warning_once
    log_warning_once(
        f"DWA behavior not registered ({_e}). "
        "Ensure `dwa.py` and `animation.py` are on PYTHONPATH, "
        "or use external registration (`import dwa_behavior_methods`)."
    )
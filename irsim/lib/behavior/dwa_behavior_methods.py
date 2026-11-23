"""
dwa_behavior_methods.py — DWA как YAML-конфигурируемое поведение ir-sim.

Зависимости:
    dwa.py — DWA, Obstacle, angle_range_corrector
    irsim.lib.behavior.behavior_registry — register_behavior_class
"""

import numpy as np

from irsim.lib.behavior.behavior_registry import register_behavior_class
from irsim.util.util import log_error, vel_world2omni

from .dwa_impl import DWA, Obstacle, Path, angle_range_corrector


# ---------------------------------------------------------------------------
# 1) DWA из dwa.py с правкой индексации и корректным учётом радиуса робота
# ---------------------------------------------------------------------------
class _SafeDWA(DWA):
    """DWA с предикцией препятствий, учётом габаритов и управлением задним ходом."""

    def __init__(self, samplingtime, robot_radius=0.0, obs_margin=0.0,
                 backward_penalty=1.0, goal_threshold=0.5):
        super().__init__(samplingtime)
        self.robot_radius = float(robot_radius)
        self.obs_margin = float(obs_margin)
        self.backward_penalty = float(backward_penalty)  # 1.0 = как было, 0.0 = свободный задний ход
        self.goal_threshold = float(goal_threshold)

    # ---- public --------------------------------------------------------
    def calc_input(self, g_x, g_y, state, obstacles):
        self.traj_paths.clear()
        self.traj_opt.clear()
        paths = self._make_path(state, g_x, g_y)
        opt_path = self._eval_path(paths, g_x, g_y, state, obstacles)
        self.traj_opt.append(opt_path)
        return paths, opt_path

    # ---- adaptive pre_step: не прогнозируем дальше цели ----------------
    def _make_path(self, state, g_x=None, g_y=None):
        min_w, max_w, min_v, max_v = self._calc_range_vels(state)

        pre_step = self.pre_step
        if g_x is not None and g_y is not None:
            dist = np.hypot(g_x - state.x, g_y - state.y)
            useful = dist + self.goal_threshold
            vmax = max(abs(min_v), abs(max_v), 1e-3)
            pre_step = int(np.ceil(useful / (vmax * self.samplingtime))) + 2
            pre_step = max(3, min(pre_step, self.pre_step))

        w_samples = (np.array([min_w]) if (max_w - min_w) < self.delta_ang_vel
                     else np.arange(min_w, max_w, self.delta_ang_vel))
        v_samples = (np.array([min_v]) if (max_v - min_v) < self.delta_vel
                     else np.arange(min_v, max_v, self.delta_vel))

        paths = []
        for w in w_samples:
            for v in v_samples:
                xs, ys, ths = self.simu_robot.predict_state(
                    w, v, state.x, state.y, state.th,
                    self.samplingtime, pre_step)
                paths.append(Path(xs, ys, ths, w, v))
        self.traj_paths.append(paths)
        return paths

    # ---- backward_penalty масштабирует вклад скорости назад ------------
    def _eval_path(self, paths, g_x, g_y, state, obstacles):
        neighbor_obs = self._calc_neighbor_obs(state, obstacles)
        safe_paths, scores = [], []
        for p in paths:
            score_obs = self._calc_obstacles_score(p, neighbor_obs)
            if score_obs == -float("inf"):
                continue
            v_term = self.weight_vel * p.u_v
            if p.u_v < 0:
                v_term *= self.backward_penalty        # 0 → нейтрально, 1 → как раньше
            scores.append(
                self.weight_angle * self._calc_heading_angle_score(p, g_x, g_y)
                + v_term
                + self.weight_obs * score_obs
            )
            safe_paths.append(p)
        if not scores:
            raise RuntimeError("All paths cannot avoid obstacles")
        return safe_paths[int(np.argmax(scores))]

    # ---- предикция: проверяем путь против положения препятствия В t ----
    def _calc_obstacles_score(self, path, neighbor_obs):
        score_sq = self.score_obstacle_sqrd
        base = self.robot_radius + self.obs_margin
        dt = self.samplingtime
        for i, (px, py) in enumerate(zip(path.xs, path.ys)):
            t = (i + 1) * dt
            for obs in neighbor_obs:
                ox = obs.x + getattr(obs, "vx", 0.0) * t
                oy = obs.y + getattr(obs, "vy", 0.0) * t
                d2 = (px - ox) ** 2 + (py - oy) ** 2
                if d2 < score_sq:
                    score_sq = d2
                r_eff = obs.size + base
                if d2 < r_eff * r_eff:
                    return -float("inf")
        return np.sqrt(score_sq)
# ---------------------------------------------------------------------------
# 2) Носитель состояния для DWA (поля x,y,th,u_v,u_th)
# ---------------------------------------------------------------------------
class _RobotState:
    __slots__ = ("x", "y", "th", "u_v", "u_th")

    def __init__(self, x=0.0, y=0.0, th=0.0, u_v=0.0, u_th=0.0):
        self.x, self.y, self.th = x, y, th
        self.u_v, self.u_th = u_v, u_th


# ---------------------------------------------------------------------------
# 3) Класс-поведение
# ---------------------------------------------------------------------------
class DwaBehavior:
    """DWA-поведение для ir-sim. Все параметры приходят из YAML (behavior_dict).

    Минимальный YAML:

        behavior:
          name: dwa
          target_roles: obstacle
          weight_obs: 0.1
          lim_max_vel: 1.6
          obs_margin: 0.1

    Радиус робота берётся из `object_info.radius` (как у ORCA). Если у объекта
    радиуса нет — используется 0.0 (точечная модель).
    """

    _DEFAULTS = {
        "weight_angle":     0.04,
        "weight_vel":       0.2,
        "weight_obs":       0.1,
        "pre_step":         30,
        "delta_vel":        0.02,
        "delta_ang_vel":    0.02,
        "area_dis_to_obs":  5.0,
        "score_obstacle":   2.0,
        "obs_margin":       0.10,
        "backward_penalty": 0.30,
        "goal_threshold":   0.5,
        "samplingtime":     0.1,
        "max_acc":          1.0,
        "max_ang_acc":      np.deg2rad(100.0),
        "lim_max_vel":      1.6,
        "lim_min_vel":      0.0,
        "lim_max_ang_vel":  np.pi,
        "lim_min_ang_vel": -np.pi,
    }

    def __init__(self, object_info, **kwargs):
        self.object_info = object_info
        self._kinematics = getattr(object_info, "kinematics", "diff")
        self._samplingtime = float(kwargs.get("samplingtime", 0.1))

        # радиус робота — как в OrcaGroupBehavior (_realizable_speed/agent)
        robot_radius = float(getattr(object_info, "radius", 0.0) or 0.0)
        obs_margin = float(kwargs.get("obs_margin", self._DEFAULTS["obs_margin"]))

        self._goal_threshold = float(
            kwargs.get("goal_threshold", self._DEFAULTS["goal_threshold"])
        )
        backward_penalty = float(
            kwargs.get("backward_penalty", self._DEFAULTS["backward_penalty"])
        )

        self._dwa = _SafeDWA(
            self._samplingtime,
            robot_radius=robot_radius,
            obs_margin=obs_margin,
            backward_penalty=backward_penalty,
            goal_threshold=self._goal_threshold,
        )
        self._apply_config(kwargs)
        # обратная связь по скорости (для динамического окна)
        self._last_v = 0.0
        self._last_w = 0.0

        self._state = _RobotState()

    # ------------------------------------------------------------------
    def _apply_config(self, cfg):
        dwa = self._dwa
        for attr in ("pre_step", "delta_vel", "delta_ang_vel",
                     "weight_angle", "weight_vel", "weight_obs"):
            setattr(dwa, attr, cfg.get(attr, getattr(dwa, attr)))

        dwa.area_dis_to_obs_sqrd = float(
            cfg.get("area_dis_to_obs", self._DEFAULTS["area_dis_to_obs"])) ** 2
        dwa.score_obstacle_sqrd = float(
            cfg.get("score_obstacle", self._DEFAULTS["score_obstacle"])) ** 2

        # obs_margin уже применён в конструкторе _SafeDWA; здесь не трогаем.

        cs = dwa.simu_robot
        for attr in ("max_acc", "max_ang_acc",
                     "lim_max_vel", "lim_min_vel",
                     "lim_max_ang_vel", "lim_min_ang_vel"):
            setattr(cs, attr, cfg.get(attr, getattr(cs, attr)))

    # ------------------------------------------------------------------
    def __call__(self, ego_object, external_objects=None, **kwargs):
        if external_objects is None:
            external_objects = []

        # ---- state ----
        state = np.asarray(ego_object.state).ravel()
        if state.size < 3:
            log_error("dwa: ego state must be [x, y, theta, ...]")
            return np.zeros((2, 1))
        x, y, th = float(state[0]), float(state[1]), float(state[2])

        # ---- goal ----
        if ego_object.goal is None:
            ego_object.logger.warning_once(
                f"{ego_object.name}: goal is None, DWA is waiting for a goal")
            return self._zero()

        goal = np.asarray(ego_object.goal).ravel()
        g_x, g_y = float(goal[0]), float(goal[1])

        # ---- step_time ----
        dt = float(getattr(ego_object._world_param, "step_time", self._samplingtime))
        if dt > 0:
            self._dwa.samplingtime = dt

        # ---- goal reached ----
        gt = getattr(ego_object, "goal_threshold", None)
        threshold = float(gt) if gt is not None else self._goal_threshold
        if (g_x - x) ** 2 + (g_y - y) ** 2 < threshold ** 2:
            self._last_v = self._last_w = 0.0
            return self._zero()

        # ---- obstacles ----
        obstacles = self._to_obstacles(external_objects)

        # ---- update DWA state ----
        self._state.x, self._state.y, self._state.th = x, y, th
        self._state.u_v, self._state.u_th = self._last_v, self._last_w

        # ---- compute ----
        try:
            _, opt_path = self._dwa.calc_input(g_x, g_y, self._state, obstacles)
            v, w = opt_path.u_v, opt_path.u_th
        except RuntimeError:
            v, w = self._emergency_control(obstacles)

        # ---- clip to robot velocity range ----
        try:
            min_step, max_step = ego_object.get_vel_range()
            v = float(np.clip(v, min_step[0, 0], max_step[0, 0]))
            if min_step.shape[0] > 1:
                w = float(np.clip(w, min_step[1, 0], max_step[1, 0]))
        except Exception:
            pass

        self._last_v, self._last_w = v, w
        return self._format_velocity(v, w, th)

    # ------------------------------------------------------------------
    def _to_obstacles(self, external_objects, spacing=0.4):
         out = []
         for obj in external_objects:
             segs = getattr(obj, "rvo_line_segments", None)
             if segs:
                for seg in segs:
                    x1, y1, x2, y2 = seg
                    L = float(np.hypot(x2 - x1, y2 - y1))
                    n = max(1, int(np.ceil(L / spacing)))
                    for i in range(n):
                        t = (i + 0.5) / n
                        cx = x1 + t * (x2 - x1)
                        cy = y1 + t * (y2 - y1)
                        out.append(Obstacle(cx, cy, spacing * 0.5))
                continue
             try:
                ox, oy, vx, vy, r = obj.rvo_neighbor_state
             except (AttributeError, TypeError, ValueError):
                continue
             out.append(Obstacle(float(ox), float(oy), float(r),
                            vx=float(vx), vy=float(vy)))
         return out

    def _format_velocity(self, v, w, th):
        if self._kinematics == "omni":
            world_v = np.array([[v * np.cos(th)], [v * np.sin(th)]])
            return vel_world2omni(th, world_v)
        return np.array([[v], [w]])

    def _zero(self):
        return np.zeros((2, 1))

    def _emergency_control(self, obstacles):
        if not obstacles:
            return 0.0, 0.0
        nearest = min(obstacles,
                      key=lambda o: (o.x - self._state.x) ** 2 + (o.y - self._state.y) ** 2)
        away = np.arctan2(self._state.y - nearest.y, self._state.x - nearest.x)
        dth = angle_range_corrector(away - self._state.th)
        lim = self._dwa.simu_robot.lim_max_ang_vel
        return 0.0, float(np.clip(1.5 * dth, -lim, lim))


# ---------------------------------------------------------------------------
# 4) Регистрация (по образцу @register_behavior("diff", "rvo"))
# ---------------------------------------------------------------------------
@register_behavior_class("diff", "dwa")
def beh_diff_dwa(object_info, **kwargs):
    return DwaBehavior(object_info, **kwargs)


@register_behavior_class("omni", "dwa")
def beh_omni_dwa(object_info, **kwargs):
    return DwaBehavior(object_info, **kwargs)
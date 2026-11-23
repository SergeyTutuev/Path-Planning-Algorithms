"""
This file is the implementation of the Reciprocal Velocity Obstacle (RVO) algorithm for multi-robot collision avoidance.

Author: Ruihua Han

reference: https://github.com/MengGuo/RVO_Py_MAS
"""

from math import asin, atan2, cos, pi, sin, sqrt, tan
import numpy as np

from irsim.util.util import dist_hypot, log_error


class reciprocal_vel_obs:
    """
    Reciprocal Velocity Obstacle (RVO) для multi-robot collision avoidance.

    Args:

        state (list): rvo state [x, y, vx, vy, radius, vx_des, vy_des].
        obs_state_list (list): circles [[x, y, vx, vy, radius], ...] (или [x,y,r]).
        vxmax (float): Maximum velocity in the x direction.
        vymax (float): Maximum velocity in the y direction.
        acce (float): Acceleration limit.
        factor (float): Penalty weighting factor for velocity selection.
        line_obs_list (list): Line segments [[x1, y1, x2, y2], ...].
        rad_approx (float | None): NEW — если задано, каждый сегмент из
            line_obs_list покрывается цепочкой окружностей этого радиуса,
            а исходные сегменты в VO-конусах больше не участвуют.
        rad_approx_overlap (float): NEW — доля перекрытия соседних кругов.
            0.0 — круги касаются, 0.5 — перекрываются на половину радиуса,
            1.0 — полностью дублируются.
        rad_approx_range (float): NEW — отсечка кругов по дистанции до ego.
            Круги дальше `rad_approx_range + ego_radius` не создают конусов.
    """

    def __init__(
    self, state, obs_state_list=None,
    vxmax=1.5, vymax=1.5, acce=0.5, factor=1.0,
    line_obs_list=None, decel=None,
    allow_backward=True, backward_penalty=0.0, save_marg=0.0,
    rad_approx=None, rad_approx_overlap=0.5, rad_approx_range=10.0,
    kinematics="diff", wheelbase=None, delta_max=None, ego_theta=None,
    idle_penalty: float = 0.0, idle_grace: float = 3.0,
    idle_penalty_cap: float = 10.0, idle_time: float = 0.0):
        if obs_state_list is None:
            obs_state_list = []
        if line_obs_list is None:
            line_obs_list = []
        self.state = state
        self.obs_state_list = obs_state_list
        self.line_obs_list = line_obs_list
        self.vxmax = vxmax
        self.vymax = vymax
        self.acce = acce
        self.decel = acce if decel is None else decel
        self.factor = factor
        self.allow_backward = allow_backward
        self.backward_penalty = backward_penalty
        self.save_marg = save_marg

        self.rad_approx = rad_approx
        self.rad_approx_overlap = float(rad_approx_overlap)
        self.rad_approx_range = float(rad_approx_range)
        self._line_circles = self._decompose_lines(self.line_obs_list)
        self.kinematics = kinematics
        self.wheelbase = wheelbase
        self.delta_max = delta_max
        self.ego_theta = ego_theta
        self.idle_penalty = float(idle_penalty)
        self.idle_grace = float(idle_grace)
        self.idle_penalty_cap = float(idle_penalty_cap)
        self.idle_time = float(idle_time)

    def update(self, state, obs_state_list, line_obs_list=None):
        self.state = state
        self.obs_state_list = obs_state_list
        self.line_obs_list = line_obs_list if line_obs_list is not None else []
        self._line_circles = self._decompose_lines(self.line_obs_list)   # NEW

    # --- NEW: circle approximation of line obstacles -------------------
    def _uses_line_circles(self) -> bool:
        return self.rad_approx is not None and self.rad_approx > 0

    def _decompose_lines(self, segments):
        """Разложить сегменты в цепочку кругов радиуса rad_approx."""
        if not segments or not self._uses_line_circles():
            return []
        rad = float(self.rad_approx)
        spacing = rad * max(2.0 - self.rad_approx_overlap, 0.1)
        circles = []
        for seg in segments:
            x1, y1, x2, y2 = seg
            L = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if L < 1e-6:
                continue
            num = max(1, int(np.ceil(L / spacing)))
            for j in range(num):
                t = (j + 0.5) / num
                circles.append([x1 + t * (x2 - x1),
                                y1 + t * (y2 - y1),
                                0.0, 0.0, rad])
        return circles

    def _circles_for_cones(self):
        """Пользовательские круги + line-круги в радиусе rad_approx_range."""
        x, y = self.state[0], self.state[1]
        rng = float(self.rad_approx_range) + self._effective_radius()
        rng_sq = rng * rng
        out = list(self.obs_state_list)
        for c in self._line_circles:
            if (c[0] - x) ** 2 + (c[1] - y) ** 2 < rng_sq:
                out.append(c)
        return out

    # --- аксессоры для оверлея ---
    def get_line_circles(self):
        """Все line-круги без фильтра по дистанции."""
        return [tuple(c) for c in self._line_circles]

    def get_active_line_circles(self):
        """Только те line-круги, что реально попадают в радиус на этом шаге."""
        if not self._line_circles:
            return []
        x, y = self.state[0], self.state[1]
        rng = float(self.rad_approx_range) + self._effective_radius()
        rng_sq = rng * rng
        return [tuple(c) for c in self._line_circles
               if (c[0] - x) ** 2 + (c[1] - y) ** 2 < rng_sq]

    def _acker_feasible_mask(self, vx, vy):
        """True там, где (vx, vy) реализуемо Ackermann-роботом.

        В body-frame: v_lat/v_fwd ≤ tan(δ_max). Нулевая скорость разрешена.
        Если δ_max ≥ π/2 — ограничения нет (эквивалент голономной).
        """
        if self.kinematics != "acker" or self.delta_max is None or self.ego_theta is None:
            return np.ones(vx.shape, dtype=bool)
        dm = float(self.delta_max)
        if dm >= pi / 2 - 1e-3:
            return np.ones(vx.shape, dtype=bool)

        th = float(self.ego_theta)
        c, s = cos(th), sin(th)
        v_fwd = vx * c + vy * s
        v_lat = -vx * s + vy * c
        max_ratio = tan(dm)
        return np.abs(v_lat) <= max_ratio * np.abs(v_fwd) + 1e-3
    
    def _effective_radius(self):
        """Радиус робота с учётом запаса безопасности save_marg."""
        return self.state[4] + self.save_marg

    # ------------------------------------------------------------------
    # Конусы: заменяем self.obs_state_list на self._circles_for_cones()
    # ------------------------------------------------------------------
    def cal_vel(self, mode="rvo"):
        if mode == "rvo":
            rvo_list = self.config_rvo()
        elif mode == "hrvo":
            rvo_list = self.config_hrvo()
        elif mode == "vo":
            rvo_list = self.config_vo()
        else:  # pragma: no cover
            log_error("wrong method mode, please input vo, rvo or hrvo")
            rvo_list = []

        velocities, outside = self._candidate_velocities(rvo_list)
        return self.vel_select(velocities[outside], velocities[~outside])

    @staticmethod
    def _cone_angles(x, y, r, mx, my, mr):
        dis_mr = np.sqrt((my - y) ** 2 + (mx - x) ** 2)
        angle_mr = atan2(my - y, mx - x)
        if dis_mr < r + mr:
            dis_mr = r + mr
        ratio = min(max((r + mr) / dis_mr, -1.0), 1.0)
        half_angle = asin(ratio)
        return angle_mr + half_angle, angle_mr - half_angle, half_angle

    @staticmethod
    def _neighbour_state(obstacle):
        if len(obstacle) == 3:
            x, y, r = obstacle
            return x, y, 0.0, 0.0, r
        return obstacle[0], obstacle[1], obstacle[2], obstacle[3], obstacle[4]

    def config_rvo(self):
        rvo_list = []
        for obstacle in self._circles_for_cones():              # NEW
            rvo_list.append(self.config_rvo_mode(obstacle))
        if not self._uses_line_circles():                        # NEW
            rvo_list.extend(self.config_vo_lines())
        return rvo_list

    def config_rvo_mode(self, obstacle):
        x = self.state[0]; y = self.state[1]
        vx = self.state[2]; vy = self.state[3]
        r = self._effective_radius()
        mx, my, mvx, mvy, mr = self._neighbour_state(obstacle)
        rvo_apex = [(vx + mvx) / 2, (vy + mvy) / 2]
        line_left_ori, line_right_ori, _ = self._cone_angles(x, y, r, mx, my, mr)
        line_left_vector = [cos(line_left_ori), sin(line_left_ori)]
        line_right_vector = [cos(line_right_ori), sin(line_right_ori)]
        return [rvo_apex, line_left_vector, line_right_vector]

    def config_hrvo(self):
        hrvo_list = []
        for obstacle in self._circles_for_cones():              # NEW
            hrvo_list.append(self.config_hrvo_mode(obstacle))
        if not self._uses_line_circles():                        # NEW
            hrvo_list.extend(self.config_vo_lines())
        return hrvo_list

    def config_hrvo_mode(self, obstacle):
        x = self.state[0]; y = self.state[1]
        vx = self.state[2]; vy = self.state[3]
        r = self._effective_radius()
        mx, my, mvx, mvy, mr = self._neighbour_state(obstacle)
        rvo_apex = [(vx + mvx) / 2, (vy + mvy) / 2]
        vo_apex = [mvx, mvy]
        line_left_ori, line_right_ori, half_angle = self._cone_angles(x, y, r, mx, my, mr)
        line_left_vector = [cos(line_left_ori), sin(line_left_ori)]
        line_right_vector = [cos(line_right_ori), sin(line_right_ori)]
        cl_vector = [mx - x, my - y]
        cur_v = [vx - rvo_apex[0], vy - rvo_apex[1]]
        dis_rv = dist_hypot(rvo_apex[0], rvo_apex[1], vo_apex[0], vo_apex[1])
        radians_rv = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])
        diff = line_left_ori - radians_rv
        temp = pi - 2 * half_angle
        if temp == 0:
            temp = temp + 0.01
        dis_diff = dis_rv * sin(diff) / sin(temp)
        if reciprocal_vel_obs.cross_product(cl_vector, cur_v) <= 0:
            hrvo_apex = [
                rvo_apex[0] - dis_diff * cos(line_right_ori),
                rvo_apex[1] - dis_diff * sin(line_right_ori),
            ]
        else:
            hrvo_apex = [
                vo_apex[0] + dis_diff * cos(line_right_ori),
                vo_apex[1] + dis_diff * sin(line_right_ori),
            ]
        return [hrvo_apex, line_left_vector, line_right_vector]

    def config_vo(self):
        vo_list = []
        for obstacle in self._circles_for_cones():              # NEW
            vo_list.append(self.config_vo_mode(obstacle))
        if not self._uses_line_circles():                        # NEW
            vo_list.extend(self.config_vo_lines())
        return vo_list

    def config_vo_mode(self, obstacle):
        x = self.state[0]; y = self.state[1]
        r = self._effective_radius()
        mx, my, mvx, mvy, mr = self._neighbour_state(obstacle)
        vo_apex = [mvx, mvy]
        line_left_ori, line_right_ori, _ = self._cone_angles(x, y, r, mx, my, mr)
        line_left_vector = [cos(line_left_ori), sin(line_left_ori)]
        line_right_vector = [cos(line_right_ori), sin(line_right_ori)]
        return [vo_apex, line_left_vector, line_right_vector]

    def config_vo_lines(self):
        vo_list = []
        x = self.state[0]; y = self.state[1]
        r = self._effective_radius()
        for seg in self.line_obs_list:
            x1, y1, x2, y2 = seg
            dx_seg = x2 - x1; dy_seg = y2 - y1
            seg_len_sq = dx_seg * dx_seg + dy_seg * dy_seg
            if seg_len_sq < 1e-12:
                continue
            t = ((x - x1) * dx_seg + (y - y1) * dy_seg) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx = x1 + t * dx_seg; cy = y1 + t * dy_seg
            dist_closest = sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist_closest < 1e-6:
                dist_closest = 1e-6
            angle1 = atan2(y1 - y, x1 - x)
            angle2 = atan2(y2 - y, x2 - x)
            dist1 = sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
            dist2 = sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
            expand1 = asin(min(1.0, r / max(dist1, r))) if dist1 > 1e-6 else pi / 2
            expand2 = asin(min(1.0, r / max(dist2, r))) if dist2 > 1e-6 else pi / 2
            diff = atan2(sin(angle2 - angle1), cos(angle2 - angle1))
            if diff >= 0:
                left_angle = angle2 + expand2
                right_angle = angle1 - expand1
            else:
                left_angle = angle1 + expand1
                right_angle = angle2 - expand2
            vo_list.append([
                [0, 0],
                [cos(left_angle), sin(left_angle)],
                [cos(right_angle), sin(right_angle)],
            ])
        return vo_list

    def vel_candidate(self, rvo_list):
        velocities, outside = self._candidate_velocities(rvo_list)
        return velocities[outside].tolist(), velocities[~outside].tolist()

    def _axis_candidates(self, current, vmax):
        if current >= 0:
            low = max(current - self.decel, -vmax)
            high = min(current + self.acce, vmax)
        else:
            low = max(current - self.acce, -vmax)
            high = min(current + self.decel, vmax)

        if high <= low:
            return np.array([float(np.clip(current, -vmax, vmax))])

        candidates = np.arange(low, high, 0.01)

        # всегда разрешаем «стоять на месте» по этой оси
        if low <= 0 <= high and not np.any(np.abs(candidates) < 1e-6):
            candidates = np.append(candidates, 0.0)
        # и всегда разрешаем сохранить текущее значение
        if low <= current <= high and not np.any(np.abs(candidates - current) < 1e-6):
            candidates = np.append(candidates, current)

        return np.unique(np.round(candidates, 6))

    def _candidate_velocities(self, rvo_list):
        vxs = self._axis_candidates(self.state[2], self.vxmax)
        vys = self._axis_candidates(self.state[3], self.vymax)
        grid_x, grid_y = np.meshgrid(vxs, vys, indexing="ij")
        vx, vy = grid_x.ravel(), grid_y.ravel()

        keep = self._acker_feasible_mask(vx, vy)
        vx, vy = vx[keep], vy[keep]

        if not self.allow_backward:
            des_x, des_y = self.state[5], self.state[6]
            des_mag = np.hypot(des_x, des_y)
            if des_mag > 1e-6:
                dot = vx * des_x + vy * des_y
                forward = dot >= -1e-9
                vx, vy = vx[forward], vy[forward]
        outside = np.ones(vx.size, dtype=bool)
        for apex, left, right, *_ in rvo_list:
            rel_x, rel_y = vx - apex[0], vy - apex[1]
            inside = (left[0] * rel_y - rel_x * left[1] <= 0) & (
                right[0] * rel_y - rel_x * right[1] >= 0
            )
            outside &= ~inside
        return np.column_stack((vx, vy)), outside

    def vo_out(self, vx, vy, rvo_list):
        for rvo in rvo_list:
            rel_vector = [vx - rvo[0][0], vy - rvo[0][1]]
            if reciprocal_vel_obs.between_vector(rvo[1], rvo[2], rel_vector):
                return False
        return True

    def vel_select(self, vo_outside, vo_inside):
        vel_des = np.array([self.state[5], self.state[6]])
        if len(vo_outside) != 0:
            velocities = np.asarray(vo_outside)
            costs = self._selection_costs(velocities, vel_des)
            return list(velocities[int(np.argmin(costs))])
        velocities = np.asarray(vo_inside)
        costs = self.penalties(velocities, vel_des, self.factor)
        return list(velocities[int(np.argmin(costs))])

    def _selection_costs(self, velocities, vel_des):
        dist = np.hypot(velocities[:, 0] - vel_des[0], velocities[:, 1] - vel_des[1])
        if self.allow_backward and self.backward_penalty > 0:
            dot = velocities[:, 0] * vel_des[0] + velocities[:, 1] * vel_des[1]
            dist = dist + self.backward_penalty * np.maximum(-dot, 0.0)
        return dist + self._idle_cost(velocities, vel_des)

    def _idle_cost(self, vels, vel_des):
        """Растущий штраф за простой: 0, пока idle_time ≤ idle_grace;
        затем растёт ∝ (idle_time − idle_grace), насыщается на idle_penalty_cap.
        Стоящий кандидат получает полный штраф, движущийся к vel_des — нулевой."""
        if self.idle_penalty <= 0.0 or self.idle_time <= self.idle_grace:
            return np.zeros(len(vels))

        ramp = min(self.idle_time - self.idle_grace, self.idle_penalty_cap)

        des_mag = float(np.hypot(vel_des[0], vel_des[1]))
        if des_mag < 1e-6:
            # нет цели — просто штрафуем за скорость ≈ 0
            speed = np.hypot(vels[:, 0], vels[:, 1])
            return self.idle_penalty * ramp * (1.0 - np.clip(speed / max(self.vxmax, 1e-6), 0.0, 1.0))

        ux, uy = vel_des[0] / des_mag, vel_des[1] / des_mag
        prog = vels[:, 0] * ux + vels[:, 1] * uy     # компонента к цели
        vmax_ref = max(self.vxmax, self.vymax, 1e-6)
        # prog ≤ 0 (стоим или назад) → коэффициент 1.0, полный штраф
        # prog = vmax → 0, штрафа нет
        return self.idle_penalty * ramp * (1.0 - np.clip(prog / vmax_ref, 0.0, 1.0))
    
    def penalties(self, vels, vel_des, factor):
        x, y, vx, vy = self.state[:4]
        r = self._effective_radius()
        tc_columns = []
        for moving in self._circles_for_cones():                # NEW
            distance = dist_hypot(moving[0], moving[1], x, y)
            diff = max(distance**2 - (r + moving[4]) ** 2, 0)
            dis_vel = np.sqrt(diff)
            trans_x = 2 * vels[:, 0] - vx - moving[2]
            trans_y = 2 * vels[:, 1] - vy - moving[3]
            tc_columns.append(dis_vel / (np.sqrt(trans_x**2 + trans_y**2) + 1e-7))
        if not self._uses_line_circles():                       # NEW
            for seg in self.line_obs_list:
                tc_columns.append(self._tc_line_segment_all(x, y, r, vels, seg))

        dist_des = np.hypot(vels[:, 0] - vel_des[0], vels[:, 1] - vel_des[1])
        if self.allow_backward and self.backward_penalty > 0:
            dot = vels[:, 0] * vel_des[0] + vels[:, 1] * vel_des[1]
            dist_des = dist_des + self.backward_penalty * np.maximum(-dot, 0.0)
        if not tc_columns:
            return dist_des + self._idle_cost(vels, vel_des)
        tc_min = np.min(tc_columns, axis=0)
        tc_min = np.where(tc_min == 0, 0.0001, tc_min)
        return factor * (1 / tc_min) + dist_des + self._idle_cost(vels, vel_des) 

    def penalty(self, vel, vel_des, factor):
        return float(self.penalties(np.array([vel], dtype=float), vel_des, factor)[0])

    @staticmethod
    def _tc_line_segment_all(x, y, r, vels, seg):
        # ... без изменений ...
        x1, y1, x2, y2 = seg
        dx_seg, dy_seg = x2 - x1, y2 - y1
        seg_len = sqrt(dx_seg * dx_seg + dy_seg * dy_seg)
        if seg_len < 1e-12:
            return np.full(len(vels), 1e6)
        nx, ny = -dy_seg / seg_len, dx_seg / seg_len
        if nx * (x - x1) + ny * (y - y1) < 0:
            nx, ny = -nx, -ny
        perp_dist = nx * (x - x1) + ny * (y - y1)
        vel_toward = -(nx * vels[:, 0] + ny * vels[:, 1])
        approaching = vel_toward > 1e-7
        tc = np.maximum((perp_dist - r) / np.where(approaching, vel_toward, 1.0), 0)
        col_x = x + vels[:, 0] * tc
        col_y = y + vels[:, 1] * tc
        t_proj = ((col_x - x1) * dx_seg + (col_y - y1) * dy_seg) / (seg_len * seg_len)
        margin = r / seg_len
        hits = approaching & (t_proj >= -margin) & (t_proj <= 1 + margin)
        return np.where(hits, tc, 1e6)

    @staticmethod
    def between_vector(line_left_vector, line_right_vector, line_vector):
        return bool(
            reciprocal_vel_obs.cross_product(line_left_vector, line_vector) <= 0
            and reciprocal_vel_obs.cross_product(line_right_vector, line_vector) >= 0
        )

    @staticmethod
    def cross_product(vector1, vector2):
        return float(vector1[0] * vector2[1] - vector2[0] * vector1[1])
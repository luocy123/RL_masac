
import math
from typing import List, Dict, Tuple
import numpy as np

from components.utils import dist, prob_jitter
from components.uav import UAV
from components.radar_static import RadarStatic
from components.radar_moving import RadarMoving
from components.striker_static import StrikerStatic
from components.striker_moving import StrikerMoving

from components.threat_snapshot import ThreatSnapshot
from components.red_comm import CommManager
from components.red_perception import RecognitionManager
from components.collision import check_collision, Rectangle, Triangle, Circle, Vector2


class TypeRepo:
    def __init__(self, cfg: dict):
        self.radar = {int(t["type_id"]): t for t in cfg["radar_types"]}
        self.striker = {int(t["type_id"]): t for t in cfg["striker_types"]}


class World:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # ===== 地图/时间 =====
        self.W, self.H = cfg["map"]["width"], cfg["map"]["height"]
        self.dt = cfg["map"]["dt"]
        self.max_ticks = cfg["map"]["max_ticks"]
        self.tick = 0

        # ===== 障碍物 =====
        self.obstacles = []
        obstacles = cfg.get("obstacles", [])
        for obs in obstacles:
            if obs["shape"] == "circle":
                cx = obs["cx"]
                cy = obs["cy"]
                r = obs["r"]
                circ = Circle(cx,cy,r)
                new_obs = {
                    **obs,
                    "entity": circ
                }
                self.obstacles.append(new_obs)
            elif obs["shape"] == "triangle":
                p1 = Vector2(obs["x1"], obs["y1"])
                p2 = Vector2(obs["x2"], obs["y2"])
                p3 = Vector2(obs["x3"], obs["y3"])
                tri = Triangle(p1, p2, p3)
                cx,cy,r = tri.get_circumcircle()
                new_obs = {
                    **obs,
                    "entity": tri,
                    "cx": cx,
                    "cy": cy,
                    "r": r
                }
                self.obstacles.append(new_obs)
            elif obs["shape"] == "rect":
                rect = Rectangle(obs["cx"] - obs["w"] / 2, obs["cy"] - obs["h"] / 2,
                                 obs["w"],
                                 obs["h"])
                cx,cy,r = rect.get_circumcircle()
                new_obs = {
                    **obs,
                    "entity": rect,
                    "r": r
                }
                self.obstacles.append(new_obs)
        self.types = TypeRepo(cfg)

        # ===== 红方 UAV 数量初始位置目标速度与独立感知/通信参数 =====
        self.uavs: List[UAV] = []
        rp_default = cfg.get("red_perception", {})
        default_sensor_range = float(rp_default.get("default_sensor_range", 60.0))
        default_comm_range = float(rp_default.get("default_comm_range", 50.0))

        self.uav_num = cfg["red_uavs"]["uav_num"]
        start_x = float(14 * np.random.rand() + 3)
        start_y = start_x
        self.spacing = cfg["red_uavs"]["spacing"]
        goal_x = cfg["red_uavs"]["goal"]["x"]
        goal_y = cfg["red_uavs"]["goal"]["y"]
        speed = cfg["red_uavs"]["speed"]
        self.start_pos = self.assign_uav_square_formation(n=self.uav_num, start_x=start_x, start_y=start_y, spacing=self.spacing)
        for i, pos in zip(range(len(self.start_pos)), self.start_pos):
            x, y = pos
            self.uavs.append(UAV(
                id=str(i),
                initial_position=np.array([x, y], dtype=float),
                goal=np.array([goal_x, goal_y], dtype=float),
                v_limit=float(speed),
                default_sensor_range=default_sensor_range,
                default_comm_range=default_comm_range
            ))


        # === 日志统计（最小化新增） =========================================
        # 每步历史、累计里程、上一步位置、结束时间（秒）
        self._hist = {u.id: [] for u in self.uavs}
        self._path_len = {u.id: 0.0 for u in self.uavs}
        self._prev_pos_log = {u.id: u.position.copy() for u in self.uavs}
        self._end_time: Dict[str, float] = {}
        # ===================================================================

        # ===== 通信与识别配置 =====
        # rp = cfg.get("red_perception", {})
        # self.use_comm = bool(rp.get("use_comm", True))
        # self.use_recog = bool(rp.get("use_recognition", True))
        # ttl_sec = float(rp.get("recognition_ttl_sec", 3.0))
        # self.ttl_ticks = max(1, int(round(ttl_sec / max(self.dt, 1e-9))))
        # self.cluster_eps = float(rp.get("cluster_eps", 5.0))

        # 类别→典型半径查表（用雷达类型库）
        self.radar_type_range_lookup: Dict[int, float] = {
            int(t["type_id"]): float(t["range"]) for t in cfg["radar_types"]
        }

        # self.comm_mgr = CommManager()
        # self.recog_mgr = RecognitionManager(
        #     ttl_ticks=self.ttl_ticks,
        #     cluster_eps=self.cluster_eps,
        #     radar_class_range_lookup=self.radar_type_range_lookup
        # )

        # 渲染辅助：通信连线 + 感知威胁圈（含动/静）
        # self.comm_links_xy: List[Tuple[np.ndarray, np.ndarray]] = []
        # self.perceived_threats: List[Tuple[np.ndarray, float]] = []             # 兼容旧渲染（中心, 半径）
        # self.perceived_threats_mob: List[Tuple[np.ndarray, float, int, str]] = []  # 新：含类别与动/静

        # ===== 雷达（记录 true class） =====
        self.radars_static: List[RadarStatic] = []
        self._radar_static_types: List[int] = []
        for r in cfg["platforms"]["radars_static"]:
            type_id = int(r["type_id"])
            t = self.types.radar[type_id]
            self.radars_static.append(
                RadarStatic(r["id"], np.array([r["x"], r["y"]], dtype=float), float(t["range"]))
            )
            self._radar_static_types.append(type_id)

        self.radars_moving: List[RadarMoving] = []
        self._radar_moving_types: List[int] = []
        for r in cfg["platforms"]["radars_moving"]:
            type_id = int(r["type_id"])
            t = self.types.radar[type_id]
            if r["mode"] == "patrol_points":
                path = [np.array([p["x"], p["y"]], dtype=float) for p in r["path"]]
                rm = RadarMoving(
                    id=r["id"], range=float(t["range"]), mode="patrol_points",
                    pos=path[0].copy(), init_pos=path[0].copy(), path=path, speed=float(r["speed"]), loop=bool(r.get("loop", True))
                )
            else:  # circle
                c = np.array([r["center"]["x"], r["center"]["y"]], dtype=float)
                rm = RadarMoving(
                    id=r["id"], range=float(t["range"]), mode="circle",
                    center=c, radius=float(r["radius"]), ang_deg_s=float(r["angular_speed_deg_s"]),
                    pos=c + np.array([float(r["radius"]), 0.0], dtype=float)
                )
            self.radars_moving.append(rm)
            self._radar_moving_types.append(type_id)

        # ===== 打击平台 =====
        self.strikers_static: List[StrikerStatic] = []
        for s in cfg["platforms"]["strikers_static"]:
            t = self.types.striker[int(s["type_id"])]
            self.strikers_static.append(
                StrikerStatic(
                    s["id"],
                    np.array([s["x"], s["y"]], dtype=float),
                    float(t["range"]),
                    float(t["base_hit_prob"])
                )
            )

        self.strikers_moving: List[StrikerMoving] = []
        for s in cfg["platforms"]["strikers_moving"]:
            t = self.types.striker[int(s["type_id"])]
            if s["mode"] == "patrol_points":
                path = [np.array([p["x"], p["y"]], dtype=float) for p in s["path"]]
                sm = StrikerMoving(
                    id=s["id"],
                    home=np.array([s["home"]["x"], s["home"]["y"]], dtype=float),
                    speed=float(s.get("speed", t["speed"])),
                    rng=float(t["range"]),
                    base_p=float(t["base_hit_prob"]),
                    mode="patrol_points",
                    path=path,
                    loop=bool(s.get("loop", True)),
                    pos=path[0].copy(),
                    init_pos=path[0].copy()
                )
            else:  # circle
                c = np.array([s["center"]["x"], s["center"]["y"]], dtype=float)
                sm = StrikerMoving(
                    id=s["id"],
                    home=np.array([s["home"]["x"], s["home"]["y"]], dtype=float),
                    speed=float(s.get("speed", t["speed"])),
                    rng=float(t["range"]),
                    base_p=float(t["base_hit_prob"]),
                    mode="circle",
                    center=c,
                    radius=float(s["radius"]),
                    ang_deg_s=float(s["angular_speed_deg_s"]),
                    pos=c + np.array([float(s["radius"]), 0.0], dtype=float),
                    init_pos=c + np.array([float(s["radius"]), 0.0], dtype=float)
                )
            self.strikers_moving.append(sm)

    # ===== 逻辑工具 =====
    def _is_active(self, u) -> bool:
        """活跃无人机：存活且未到达（降落）"""
        return u.alive and not getattr(u, "reached_goal", False)

    def alarm_targets(self) -> List[UAV]:
        """目标级报警集合（零时延：任一雷达圈内即报警），仅活跃UAV参与"""
        res = []
        for u in self.uavs:
            if not self._is_active(u):
                continue
            in_any = any(dist(rs.pos, u.position) <= rs.range for rs in self.radars_static) or \
                     any(dist(rm.pos, u.position) <= rm.range for rm in self.radars_moving)
            if in_any:
                res.append(u)
        return res

    def assign_targets(self, alarms: List[UAV]):
        """空闲移动打击平台分配最近目标（贪心；并列按 id 破局）"""
        alive_targets = [u for u in alarms if self._is_active(u)]
        if not alive_targets:
            return
        for s in self.strikers_moving:
            if s.state == "Idle":
                alive_targets.sort(key=lambda u: (dist(s.pos, u.position), u.id))
                tgt = alive_targets[0]
                s.state, s.target_id = "Chase", tgt.id
                s.home = s.pos
            elif s.state == "Return":
                alive_targets.sort(key=lambda u: (dist(s.pos, u.position), u.id))
                tgt = alive_targets[0]
                s.state, s.target_id = "Chase", tgt.id

    def _build_snapshot(self) -> ThreatSnapshot:
        radar_centers = [rs.pos.copy() for rs in self.radars_static] + \
                        [rm.pos.copy() for rm in self.radars_moving]
        radar_ranges = [float(rs.range) for rs in self.radars_static] + \
                       [float(rm.range) for rm in self.radars_moving]
        radar_classes = self._radar_static_types + self._radar_moving_types

        strike_centers = [ss.pos.copy() for ss in self.strikers_static] + \
                         [sm.pos.copy() for sm in self.strikers_moving]
        strike_ranges = [float(ss.rng) for ss in self.strikers_static] + \
                        [float(sm.rng) for sm in self.strikers_moving]

        return ThreatSnapshot(
            tick=self.tick,
            radar_centers=radar_centers,
            radar_ranges=radar_ranges,
            radar_classes=radar_classes,
            strike_centers=strike_centers,
            strike_ranges=strike_ranges
        )

    # ===== 无人机初始坐标分配 =====
    def assign_uav_square_formation(self, n, start_x=0, start_y=0, spacing=1):
        positions = []
        if n <= 0:
            return positions
        if n == 1:
            positions.append((start_x, start_y))
            return positions

        sqrt_n = math.sqrt(n)
        min_size = math.ceil(sqrt_n)
        rows, cols = min_size, min_size

        # 调整行列数以容纳所有无人机
        while rows * cols < n:
            if rows <= cols:
                rows += 1
            else:
                cols += 1

        # 计算偏移量（确保方阵以(start_x, start_y)为中心）
        total_width = (cols - 1) * spacing
        total_height = (rows - 1) * spacing
        start_offset_x = start_x - total_width / 2
        start_offset_y = start_y - total_height / 2

        # 生成所有可能位置（含冗余）
        # 关键修改：使用reversed(range(rows))使行从下到上遍历
        all_positions = []
        for i in reversed(range(rows)):  # 从下到上遍历行
            for j in range(cols):  # 从左到右遍历列
                x = start_offset_x + j * spacing
                y = start_offset_y + i * spacing
                all_positions.append((x, y))

        # 计算需跳过的冗余位置（对称跳过）
        skip_count = rows * cols - n
        skip_indices = []
        for k in range(skip_count):
            if k % 2 == 0:
                skip_indices.append(k // 2)  # 从前面跳过
            else:
                skip_indices.append(len(all_positions) - 1 - (k // 2))  # 从后面跳过

        # 筛选最终位置（按顺序保留非跳过位置）
        count = 0
        for idx, (x, y) in enumerate(all_positions):
            if idx not in skip_indices:
                positions.append((round(x, 1), round(y, 1)))
                count += 1
                if count >= n:
                    break
        return positions

    # ===== 主循环 =====
    def step(self, actions, pre_states):
        self.tick += 1
        dt = self.dt

        # 1) 红方推进
        for uav, action in zip(self.uavs, actions):
            a_norm, a_angle = action
            uav.control(a_norm, a_angle)
            uav.update(self.dt)
        # 2) 雷达推进（移动雷达位置）
        for rm in self.radars_moving:
            rm.step(dt)
        # 4) 计算报警集合
        alarms = self.alarm_targets()
        alarm_ids = {u.id for u in alarms}
        # 5) 移动打击平台状态推进（含“失联即回家”）
        from random import random
        for s in self.strikers_moving:
            if s.state == "Idle":
                s.step_idle(dt)
            elif s.state == "Chase":
                tgt = next((u for u in self.uavs if u.id == s.target_id and self._is_active(u)), None)
                if tgt is None:
                    s.state, s.target_id = "Return", None
                    continue
                # 目标不在雷达报警集合 → 视为丢失 → 回家
                if s.target_id not in alarm_ids:
                    s.state, s.target_id = "Return", None
                    continue
                s.step_line_to(dt, tgt.position)
            elif s.state == "Return":
                s.step_line_to(dt, s.home)
                if dist(s.pos, s.home) < 1e-4:
                    s.state = "Idle"
        # 7) 分配
        self.assign_targets(alarms)
        cur_states = self.get_states()
        rewards, dones = self.get_rewards(pre_states=pre_states, cur_states=cur_states)
        return cur_states, rewards, dones


    def get_states(self):
        states = []

        if self.uavs is None:
            return states
        for uav in self.uavs:
            uav.find_obs = []
            uav.find_striker = []
            uav.find_radar = []
            uav_x, uav_y = uav.get_position()
            uav_vx, uav_vy = uav.get_velocity()

            # 到达目标点距离与方位角
            target_x, target_y = uav.goal
            x = target_x - uav_x
            y = target_y - uav_y
            dis2target = math.hypot(x, y)
            angle2target = math.atan2(y, x)
            # 自身速度大小与角度
            v_norm = math.hypot(uav_vx, uav_vy)
            v_angle = math.atan2(uav_vy, uav_vx)
            # 最近边界距离
            dis_down = abs(uav_y)
            dis_up = abs(uav_y - self.H)
            dis_left = abs(uav_x)
            dis_right = abs(uav_x - self.W)
            dis2edge = min([dis_down, dis_up, dis_left, dis_right])
            angle2edge = 0.0
            if dis2edge == dis_down:
                angle2edge = -math.pi / 2
            elif dis2edge == dis_up:
                angle2edge = math.pi / 2
            elif dis2edge == dis_left:
                angle2edge = math.pi
            elif dis2edge == dis_right:
                angle2edge = 0.0
            # 最近无人机
            disto2 = float('inf')
            near_uav = None
            uavs = self.uavs
            for other_uav in uavs:
                if other_uav.id != uav.id and not other_uav.dead_flag:
                    near_x, near_y = other_uav.get_position()
                    temp = (near_x - uav_x) ** 2 + (near_y - uav_y) ** 2
                    if temp < disto2:
                        disto2 = temp
                        near_uav = other_uav
            dis2other = math.sqrt(disto2)
            if dis2other > uav.default_sensor_range or near_uav.arrival_flag:
                dis2other = uav.default_sensor_range
                if angle2target < 0:
                    angle2other = angle2target + 3.14
                else:
                    angle2other = angle2target - 3.14
            else:
                otherx, othery = near_uav.get_position()
                angle2other = math.atan2(othery - uav_y, otherx - uav_x)


            # 查找并保存在探测范围内的障碍物
            for obstacle in self.obstacles:
                ox = obstacle["cx"]
                oy = obstacle["cy"]
                r = obstacle["r"]
                temp = math.hypot(ox - uav_x, oy - uav_y)
                if temp < uav.default_sensor_range + r:
                    uav.find_obs.append([obstacle,temp])
            if uav.find_obs:
                uav.find_obs.sort(key=lambda x: x[1])
                dis2obs = uav.find_obs[0][1]
                ox = uav.find_obs[0][0]["cx"] #注意是赋值还是引用
                oy = uav.find_obs[0][0]["cy"]
                angle2obs = math.atan2(oy - uav_y, ox - uav_x)
            else:
                dis2obs = uav.default_sensor_range
                if angle2target > 0:
                    angle2obs = angle2target - math.pi
                else:
                    angle2obs = angle2target + math.pi
            # 查找并保存在探测范围内的雷达
            for radar in (self.radars_static+self.radars_moving):
                ox,oy = radar.pos
                r = radar.range
                temp = math.hypot(ox - uav_x, oy - uav_y)
                if temp < uav.default_sensor_range + r:
                    uav.find_radar.append([radar, temp])
            if uav.find_radar:
                uav.find_radar.sort(key=lambda x: x[1])
                dis2radar = uav.find_radar[0][1]
                ox, oy = uav.find_radar[0][0].pos  # 注意是赋值还是引用

                angle2radar = math.atan2(oy - uav_y, ox - uav_x)
            else:
                dis2radar = uav.default_sensor_range
                if angle2target > 0:
                    angle2radar = angle2target - math.pi
                else:
                    angle2radar = angle2target + math.pi
            # 查找并保存在探测范围内的打击平台
            for striker in (self.strikers_static+self.strikers_moving):
                ox,oy = striker.pos
                r = striker.rng
                temp = math.hypot(ox - uav_x, oy - uav_y)
                if temp < uav.default_sensor_range + r:
                    uav.find_striker.append([striker, temp])
            if uav.find_striker:
                uav.find_striker.sort(key=lambda x: x[1])
                dis2striker = uav.find_striker[0][1]
                ox,oy = uav.find_striker[0][0].pos  # 注意是赋值还是引用

                angle2striker = math.atan2(oy - uav_y, ox - uav_x)
            else:
                dis2striker = uav.default_sensor_range
                if angle2target > 0:
                    angle2striker = angle2target - math.pi
                else:
                    angle2striker = angle2target + math.pi


            state = [v_norm, v_angle, dis2edge, angle2edge,
                     dis2obs, angle2obs, dis2radar, angle2radar, dis2striker, angle2striker,
                     dis2other, angle2other, dis2target, angle2target, uav.odometer]
            states.append(state)
        return states

    def get_rewards(self, pre_states, cur_states, env_size=100):
        rewards = []
        dones = []
        for pre_state, cur_state, uav in zip(pre_states, cur_states, self.uavs):
            reward = 0.0
            if uav.dead_flag:
                done = True
                rewards.append(reward)
                dones.append(done)
                continue
            elif uav.arrival_flag:
                done = True
                rewards.append(reward)
                dones.append(done)
                continue
            elif not uav.dead_flag:
                done = False
                pre_v_norm, pre_v_angle, pre_dis2edge, pre_angle2edge, \
                pre_dis2obs, pre_angle2obs, pre_dis2radar, pre_angle2radar, pre_dis2striker, pre_angle2striker, \
                pre_dis2other, pre_angle2other, pre_dis2target, pre_angle2target, pre_odom = pre_state
                cur_v_norm, cur_v_angle, cur_dis2edge, cur_angle2edge, \
                    cur_dis2obs, cur_angle2obs, cur_dis2radar, cur_angle2radar, cur_dis2striker, cur_angle2striker, \
                    cur_dis2other, cur_angle2other, cur_dis2target, cur_angle2target, cur_odom = cur_state
                # 障碍物碰撞奖励，打击平台奖励
                if uav.find_obs:
                    obs = uav.find_obs[0][0]
                    ux,uy = uav.position
                    uav_circ = Circle(ux,uy,uav.radius)
                    if check_collision(uav_circ, obs["entity"]):
                        reward = -(50 - (self.tick * 20 / 500))
                        done = True
                        uav.dead_flag = True
                        uav.alive = False
                        # print(f"uav {uav.uav_id} dead for obs at {self.tick} tick reward is {reward} , odom is {uav.odometer}")
                        rewards.append(reward)
                        dones.append(done)
                        continue
                if uav.find_striker:
                    if cur_dis2striker < uav.find_striker[0][0].rng + uav.radius:
                        reward = -(50 - (self.tick * 20 / 500))
                        done = True
                        uav.dead_flag = True
                        uav.alive = False
                        # print(f"uav {uav.uav_id} dead for striker at {self.tick} tick reward is {reward} , odom is {uav.odometer}")
                        rewards.append(reward)
                        dones.append(done)
                        continue
                # 越界奖励
                x, y = uav.get_position()
                if x < uav.radius or x > env_size - uav.radius or y < uav.radius or y > env_size - uav.radius:
                    reward = -(50 - (self.tick * 20 / 500))
                    done = True
                    uav.dead_flag = True
                    uav.alive = False
                    # print(f"uav {uav.uav_id} dead for edge at {self.tick} tick reward is {reward} odom is {uav.odometer}")
                    rewards.append(reward)
                    dones.append(done)
                    continue
                # 无人机碰撞奖励
                if cur_dis2other < uav.radius * 2 and not uav.arrival_flag:
                    reward = -(50 - (self.tick * 20 / 500))
                    done = True
                    uav.dead_flag = True
                    uav.alive = False
                    # print(f"uav {uav.uav_id} dead for other uav at {self.tick} tick reward is {reward} odom is {uav.odometer}")
                    rewards.append(reward)
                    dones.append(done)
                    continue
                # 到达目标奖励
                if cur_dis2target < uav.goal_radius:
                    reward = (50 - self.tick * 20 / 500)
                    done = True
                    uav.arrival_flag = True
                    # print(f"uav {uav.uav_id} successful arrival at {self.tick} tick reward is {reward}")
                    rewards.append(reward)
                    dones.append(done)
                    continue
                # 目标距离奖励
                if pre_dis2target < cur_dis2target:
                    reward -= 10.0 * cur_dis2target / self.W + 10.0
                else:
                    reward += 10.0
                # 目标角度奖励
                # pre_angle_error = abs(pre_v_angle - pre_angle2target)
                cur_angle_error = abs(cur_v_angle - cur_angle2target)
                if cur_angle_error > 0.1:
                    reward -= 4.0 * cur_angle_error + 4.0
                else:
                    reward += 2.0

                # 边界距离奖励
                if cur_dis2edge < 2.0:
                    reward -= 1.0 * (2.0 - cur_dis2edge) + 1.0
                    # 边界角度奖励
                    cur_angle_error = abs(cur_v_angle - cur_angle2edge)
                    if cur_angle_error < 1.57:
                        reward -= 0.5 * (3.14 - cur_angle_error) + 0.5

                # 障碍物距离奖励
                if cur_dis2obs < uav.default_sensor_range:
                    reward -= 2.0 * (uav.default_sensor_range - cur_dis2obs)/uav.default_sensor_range + 2.0
                    # 障碍物角度奖励
                    # pre_angle_error = abs(pre_v_angle - pre_angle2obs)
                    cur_angle_error = abs(cur_v_angle - cur_angle2obs)
                    if cur_angle_error < 1.57:
                        reward -= 1.0 * (3.14 - cur_angle_error) + 1.0
                # 雷达距离奖励
                if cur_dis2radar < uav.default_sensor_range:
                    r = uav.find_radar[0][0].range
                    if cur_dis2radar < r + uav.radius:
                        reward -= 1.0 * (r-cur_dis2radar) / r + 1.0
                    else:
                        reward -= 0.5
                    cur_angle_error = abs(cur_v_angle - cur_angle2radar)
                    if cur_angle_error < 1.57:
                        reward -= 0.5 * (3.14 - cur_angle_error) + 0.5
                #打击平台距离奖励
                if cur_dis2striker < uav.default_sensor_range:
                    r = uav.find_striker[0][0].rng
                    reward -= 2.0 * (uav.default_sensor_range - cur_dis2obs) / uav.default_sensor_range + 2.0
                    cur_angle_error = abs(cur_v_angle - cur_angle2striker)
                    if cur_angle_error < 1.57:
                        reward -= 1.0 * (3.14 - cur_angle_error) + 1.0
                # 最近无人机距离奖励
                if cur_dis2other < uav.default_sensor_range:
                    if cur_dis2other < uav.radius * 2 + 0.5:
                        reward -= 1.0 * cur_dis2other + 1.0
                    # 最近无人机角度奖励
                    cur_angle_error = abs(cur_v_angle - cur_angle2other)
                    if cur_angle_error < 1.57:
                        reward -= 0.5 * (1.57 - cur_angle_error) + 0.5

                # 里程计奖励
                reward -= 0.4 * cur_odom / self.W
                reward = reward / 25.0
                rewards.append(reward)
                dones.append(done)
        normalized_rewards = []
        for i, reward in zip(range(len(rewards)), rewards):
            if reward <= -25 or reward >= 25:
                normalized_rewards.append(reward)
            else:
                # 第一步：预缩放，将敏感区[-25, 0]映射到tanh敏感区[-1, 1]
                scaled = reward
                # 第二步：应用tanh非线性变换
                normalized = np.tanh(scaled)
                normalized_rewards.append(normalized)
                # print(f"uav {i} at {self.tick} tick reward is {normalized}")

        return normalized_rewards, dones

    def reset(self):
        start_x = float(14 * np.random.rand() + 3)
        start_y = start_x

        self.start_pos = self.assign_uav_square_formation(n=self.uav_num, start_x=start_x, start_y=start_y, spacing=self.spacing)
        for u,start in zip(self.uavs,self.start_pos):

            u.reset()
            u.position = np.array(start)
        for rm in self.radars_moving:
            rm.reset()
        for sm in self.strikers_moving:
            sm.reset()
        self.tick = 0
        return self.get_states()


    def done(self) -> bool:
        if self.tick >= self.max_ticks:
            return True
        return all((not u.alive) or u.arrival_flag for u in self.uavs)

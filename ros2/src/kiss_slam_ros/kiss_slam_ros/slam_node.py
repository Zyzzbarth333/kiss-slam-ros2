import rclpy
from rclpy.logging import get_logger
import numpy as np
import math
import message_filters
import time
import functools
import os
import json
import hashlib
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger
import std_msgs.msg
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from kiss_icp.voxelization import voxel_down_sample
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.loop_closer import LoopCloser
from kiss_slam.pose_graph_optimizer import PoseGraphOptimizer
from kiss_slam.voxel_map import VoxelMap

from kiss_slam_ros.utils.config import declare_parameters, get_kiss_slam_config

# B1: PGO information matrix weighting
_ODOM_INFORMATION = np.eye(6)
_CLOSURE_INFORMATION = 10.0 * np.eye(6)


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        if args and hasattr(args[0], 'get_logger') and callable(args[0].get_logger):
            args[0].get_logger().info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        else:
            print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # Parameters
        declare_parameters(self)
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.config = get_kiss_slam_config(self)

        # C1: Map save (map_save_directory already declared by declare_parameters)
        from datetime import datetime
        base_dir = self.get_parameter('map_save_directory').value
        run_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._save_dir = os.path.join(base_dir, run_stamp)

        # Remote sync — rsync session folder to laptop on shutdown
        # Tries each destination in order until one succeeds (ethernet, wifi, etc)
        self.declare_parameter('sync_destinations', [
            'ijziebarth@192.168.10.105:~/theseus_autonomy_ws/data/maps/hardware/',
        ])
        self._sync_destinations = self.get_parameter('sync_destinations').value or []

        # Arena config (for composite map + mission report)
        self.declare_parameter('arena_config_path', '')
        arena_path = self.get_parameter('arena_config_path').value
        self._arena_config = self._load_arena_config(arena_path)
        if self._arena_config:
            n_placards = len(self._arena_config.get('placards', []))
            self.get_logger().info(
                f"Arena config loaded: {arena_path} ({n_placards} placards)"
            )
        else:
            self.get_logger().warn(
                f"Arena config NOT loaded (path='{arena_path}')"
            )
        # Compute arena bounds
        if self._arena_config and 'arena' in self._arena_config:
            a = self._arena_config['arena']
            # Prefer pre-computed nav_bounds if present
            nb = a['nav_bounds']
            a['x_min'] = float(nb['x_min'])
            a['x_max'] = float(nb['x_max'])
            a['y_min'] = float(nb['y_min'])
            a['y_max'] = float(nb['y_max'])
            self.get_logger().info(
                f"Arena bounds: X=[{a['x_min']}, {a['x_max']}] "
                f"Y=[{a['y_min']}, {a['y_max']}]"
            )

        # Detection accumulation (from perception nodes)
        self._aruco_detections = {}   # id -> {'x': float, 'y': float, 'z': float}
        self._cube_detections = {}    # color_id -> {'x','y','z','label'}
        self._robot_path_xy = []      # [(x, y), ...] sampled from odom (map frame)
        self._flio_path_xy = []       # [(x, y), ...] raw FLIO poses (camera_init frame)
        self._path_sample_dist = 0.2  # metres between path samples
        self._start_time = time.time()

        # Publish parameters
        self.declare_parameter('publish.scan_publish_en', True)
        self.declare_parameter('publish.dense_publish_en', False)

        self.scan_pub_en = self.get_parameter('publish.scan_publish_en').value
        self.dense_pub_en = self.get_parameter('publish.dense_publish_en').value

        # Core SLAM Backend
        self.closer = LoopCloser(self.config.loop_closer)
        local_map_config = self.config.local_mapper
        self.local_map_voxel_size = local_map_config.voxel_size
        self.voxel_grid = VoxelMap(self.local_map_voxel_size)
        self.odom_local_map = VoxelMap(self.local_map_voxel_size)
        self.local_map_graph = LocalMapGraph()
        self.local_map_splitting_distance = local_map_config.splitting_distance
        self.optimizer = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        self.closures = []

        # Initial PG variable + anchor
        self.optimizer.add_variable(
            self.local_map_graph.last_id,
            self.local_map_graph.last_keypose
        )
        self.optimizer.fix_variable(self.local_map_graph.last_id)

        # State variables
        self.local_maps = []
        self.voxel_maps = []          # A2: numpy arrays, not Open3D PCDs
        self.last_split_pose = np.eye(4)

        # map→odom TF
        self._current_odom_pose = np.eye(4)
        self._map_T_odom = np.eye(4)

        # A1: Global map cache
        self._cached_global_map = np.empty((0, 3), dtype=np.float64)
        self._cached_map_count = 0
        self._cache_dirty = False

        # B2: Stable occupancy grid origin
        self._og_origin = None
        self._og_size = None

        # Occupancy grid cache — xxhash gating to skip unchanged rebuilds
        self._last_map_hash = None
        self._last_occupancy_msg = None

        # C2: SLAM status
        self._total_travel_m = 0.0
        self._last_correction_norm = 0.0
        self._last_pose_for_travel = None

        # ROS Communications
        self.fast_callback_group = ReentrantCallbackGroup()
        self.slow_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._init_publishers()
        self._init_subscribers()
        self._init_services()

        self.get_logger().info(
            f'KISS-SLAM configuration:\n'
            f'  Map Frame: {self.map_frame}\n'
            f'  Odometry Frame: {self.odom_frame}\n'
            f'  Local Mapper: {self.config.local_mapper}\n'
            f'  Loop Closer: {self.config.loop_closer}\n'
            f'  PGO: {self.config.pose_graph_optimizer}\n'
            f'  scan_publish_en: {self.scan_pub_en}\n'
            f'  dense_publish_en: {self.dense_pub_en}\n'
            f'  Map Save Dir: {self._save_dir}\n'
        )
        self.get_logger().info("SLAM node initialized and waiting for keyframes.")

    # =====================================================================
    # ROS setup
    # =====================================================================
    def _init_publishers(self):
        qos = QoSProfile(durability=DurabilityPolicy.VOLATILE,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        map_qos = QoSProfile(durability=DurabilityPolicy.VOLATILE,
                             reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pose_pub = self.create_publisher(PoseStamped, 'global_pose', qos)
        self.global_voxel_map_pub = self.create_publisher(PointCloud2, 'global_voxel_map', qos)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.cloud_registered_pub = self.create_publisher(PointCloud2, '/cloud_registered', qos)

        # C2: SLAM status
        self.status_pub = self.create_publisher(String, '/slam_status', qos)

        # Occupancy grid for Nav2 global planner (10s interval)
        self.create_timer(10.0, self.publish_2D_map, callback_group=self.slow_callback_group)
        self.create_timer(2.0, self._publish_status, callback_group=self.slow_callback_group)

    def _init_subscribers(self):
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE, depth=10)
        deskewed_points_sub = Subscriber(self, PointCloud2, 'deskewed_points', qos_profile=qos)
        odom_sub = Subscriber(self, Odometry, 'odometry', qos_profile=qos)
        self.ts = ApproximateTimeSynchronizer(
            [deskewed_points_sub, odom_sub], queue_size=20, slop=0.05
        )
        self.ts.registerCallback(self.keyframe_callback)

        # Detection listeners (RELIABLE — published by perception nodes)
        rel_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE, depth=5)
        self.create_subscription(
            MarkerArray, '/aruco_visualisation', self._aruco_cb, rel_qos)
        self.create_subscription(
            MarkerArray, '/cube_markers', self._cube_cb, rel_qos)

        # Path tracking from odom
        self.create_subscription(
            Odometry, '/odometry_ground', self._path_cb, qos)

    def _init_services(self):
        self.create_service(Trigger, '/save_map', self._save_map_callback)

    # =====================================================================
    # Arena config + detection callbacks
    # =====================================================================
    @staticmethod
    def _load_arena_config(path):
        """Load arena config YAML (placard positions, bounds)."""
        if not path or not os.path.isfile(path):
            return None
        try:
            import yaml
            with open(path) as f:
                raw = yaml.safe_load(f)
            # Unwrap ROS2 parameter nesting: /**/ros__parameters/...
            if '/**' in raw and 'ros__parameters' in raw['/**']:
                return raw['/**']['ros__parameters']
            if 'ros__parameters' in raw:
                return raw['ros__parameters']
            return raw
        except Exception:
            return None

    def _aruco_cb(self, msg: MarkerArray):
        for m in msg.markers:
            self._aruco_detections[m.id] = {
                'x': m.pose.position.x,
                'y': m.pose.position.y,
                'z': m.pose.position.z,
            }

    def _cube_cb(self, msg: MarkerArray):
        for m in msg.markers:
            # Only ingest actual cube geometry (ns='cubes'), skip confidence
            # rings, text labels, and any other auxiliary namespaces.
            if m.ns not in ('cubes', 'cube_markers'):
                continue
            # Skip non-solid marker types (text labels, line strips, etc.)
            if m.type not in (1, 2, 3):  # CUBE=1, SPHERE=2, CYLINDER=3
                continue
            label = self._colour_from_rgba(m.color.r, m.color.g, m.color.b)
            self._cube_detections[m.id] = {
                'x': m.pose.position.x,
                'y': m.pose.position.y,
                'z': m.pose.position.z,
                'label': label,
                'r': m.color.r, 'g': m.color.g, 'b': m.color.b,
            }

    @staticmethod
    def _colour_from_rgba(r, g, b):
        """Infer cube colour name from marker RGBA values."""
        if r > 0.6 and g < 0.3 and b < 0.3:
            return 'red'
        if g > 0.6 and r < 0.3 and b < 0.3:
            return 'green'
        if b > 0.6 and r < 0.3 and g < 0.3:
            return 'blue'
        if r > 0.6 and g > 0.6 and b > 0.6:
            return 'white'
        return f'cube'

    def _path_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Raw FLIO pose (camera_init/odom frame, for PCD alignment)
        if (not self._flio_path_xy or
                math.sqrt((x - self._flio_path_xy[-1][0]) ** 2 +
                           (y - self._flio_path_xy[-1][1]) ** 2) > self._path_sample_dist):
            self._flio_path_xy.append((x, y))

        # Map-corrected path (for presentation overlay)
        corrected = self._map_T_odom @ self._current_odom_pose
        cx, cy = float(corrected[0, 3]), float(corrected[1, 3])
        if self._robot_path_xy:
            dx = cx - self._robot_path_xy[-1][0]
            dy = cy - self._robot_path_xy[-1][1]
            if (dx * dx + dy * dy) < self._path_sample_dist ** 2:
                return
        self._robot_path_xy.append((cx, cy))

    # =====================================================================
    # Presentation map — FLIO PCD primary, slide-ready
    # =====================================================================
    def _generate_presentation_map(self):
        """Render presentation map from FLIO PCD with detection overlays."""
        import signal
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            self.get_logger().warn("matplotlib not available")
            return

        # Load FLIO PCD (must be copied to session dir before this runs)
        flio_pcd_path = os.path.join(self._save_dir, 'flio_map.pcd')
        if not os.path.exists(flio_pcd_path):
            self.get_logger().warn(
                f"FLIO PCD not found at {flio_pcd_path} — skipping presentation map")
            return

        pcd = self._read_pcd_numpy(flio_pcd_path)
        if pcd is None or pcd.shape[0] < 100:
            self.get_logger().warn(
                f"FLIO PCD too small ({0 if pcd is None else pcd.shape[0]} pts)")
            return

        self.get_logger().info(
            f"Building presentation map from FLIO PCD: {pcd.shape[0]:,} points")

        # Arena clip
        pad = 2.0
        cfg = self._arena_config
        if cfg and 'arena' in cfg:
            a = cfg['arena']
            mask = (
                (pcd[:, 0] >= a.get('x_min', -20) - pad) &
                (pcd[:, 0] <= a.get('x_max', 20) + pad) &
                (pcd[:, 1] >= a.get('y_min', -20) - pad) &
                (pcd[:, 1] <= a.get('y_max', 20) + pad))
            pcd = pcd[mask]
            self.get_logger().info(
                f"Arena clip: {mask.sum():,} / {len(mask):,} points kept")

        if pcd.shape[0] < 100:
            self.get_logger().warn("Not enough points after arena clip")
            return

        # 2D projection at 0.05m
        res = 0.05
        xy = pcd[:, :2]
        x_min = float(np.floor(np.min(xy[:, 0]) - 1))
        x_max = float(np.ceil(np.max(xy[:, 0]) + 1))
        y_min = float(np.floor(np.min(xy[:, 1]) - 1))
        y_max = float(np.ceil(np.max(xy[:, 1]) + 1))

        nx = int((x_max - x_min) / res) + 1
        ny = int((y_max - y_min) / res) + 1
        cx = ((pcd[:, 0] - x_min) / res).astype(int)
        cy = ((pcd[:, 1] - y_min) / res).astype(int)
        valid = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        cx, cy = cx[valid], cy[valid]
        zvals = pcd[valid, 2]
        flat = cy * nx + cx

        max_z = np.full(nx * ny, np.nan, dtype=np.float64)
        np.maximum.at(max_z, flat, zvals)
        counts = np.zeros(nx * ny, dtype=np.int32)
        np.add.at(counts, flat, 1)

        valid_z = max_z[np.isfinite(max_z)]
        if len(valid_z) == 0:
            return
        ground_z = float(np.median(valid_z)) + 0.10
        z_range = max(float(valid_z.max()) - ground_z, 0.5)

        # Build RGB image
        img = np.full((ny, nx, 3), 40, dtype=np.uint8)
        observed = counts > 0
        for idx in np.where(observed)[0]:
            z = max_z[idx]
            r, c = divmod(idx, nx)
            if z < ground_z:
                img[r, c] = [220, 220, 220]
            else:
                t = np.clip((z - ground_z) / z_range, 0, 1)
                if t < 0.25:
                    s = t / 0.25
                    img[r, c] = [0, int(s * 200), 200]
                elif t < 0.5:
                    s = (t - 0.25) / 0.25
                    img[r, c] = [0, 200, int(200 * (1 - s))]
                elif t < 0.75:
                    s = (t - 0.5) / 0.25
                    img[r, c] = [int(s * 255), 200, 0]
                else:
                    s = (t - 0.75) / 0.25
                    img[r, c] = [255, int(200 * (1 - s)), 0]

        img = np.flipud(img)
        extent = [x_min, x_max, y_min, y_max]

        fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi=150)
        ax.imshow(img, origin='lower', extent=extent, aspect='equal', alpha=0.9)

        # Arena boundary
        if cfg and 'arena' in cfg:
            a = cfg['arena']
            rect = Rectangle(
                (a['x_min'], a['y_min']),
                a['x_max'] - a['x_min'], a['y_max'] - a['y_min'],
                linewidth=2.5, edgecolor='red', facecolor='none',
                linestyle='--', label='Arena boundary')
            ax.add_patch(rect)

        # FLIO trajectory (camera_init frame — matches PCD perfectly)
        if len(self._flio_path_xy) > 1:
            px, py = zip(*self._flio_path_xy)
            ax.plot(px, py, '-', color='#2196F3', linewidth=2.0,
                    alpha=0.9, label='Rover trajectory')
            ax.plot(px[0], py[0], '*', color='gold', markersize=20,
                    markeredgecolor='black', markeredgewidth=1.0,
                    label='Start', zorder=10)
            ax.plot(px[-1], py[-1], 'o', color='#FF5722', markersize=12,
                    markeredgecolor='black', markeredgewidth=1.0,
                    label='End', zorder=10)

        # Cube detections
        cube_colours = {
            'blue': '#2196F3', 'green': '#4CAF50',
            'red': '#F44336', 'white': '#FFFFFF'}
        first_cube = True
        for cid, det in self._cube_detections.items():
            colour_name = det.get('colour', det.get('label', 'white')).lower()
            clr = cube_colours.get(colour_name, '#888888')
            dx, dy = det.get('x', 0), det.get('y', 0)
            ax.plot(dx, dy, 's', color=clr, markersize=16,
                    markeredgecolor='black', markeredgewidth=1.5,
                    zorder=9, label='Cube' if first_cube else '')
            first_cube = False
            ax.annotate(
                f'{colour_name.upper()}\n({dx:.2f}, {dy:.2f})',
                (dx, dy), textcoords='offset points', xytext=(14, 10),
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=clr, alpha=0.9, linewidth=2))

        # ArUco / placard detections
        first_aruco = True
        for aid, det in self._aruco_detections.items():
            ax_x, ay = det.get('x', 0), det.get('y', 0)
            ax.plot(ax_x, ay, '^', color='#FF9800', markersize=14,
                    markeredgecolor='black', markeredgewidth=1.0,
                    zorder=9, label='Placard' if first_aruco else '')
            first_aruco = False
            ax.annotate(
                f'P{aid}\n({ax_x:.2f}, {ay:.2f})',
                (ax_x, ay), textcoords='offset points', xytext=(14, -14),
                fontsize=9, fontweight='bold', color='#E65100',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9))

        # Placard config positions (expected)
        if cfg and 'placards' in cfg:
            placards = cfg['placards']
            if isinstance(placards, dict):
                placards = list(placards.get('positions', {}).values())
            first_cfg = True
            for p in placards:
                if not isinstance(p, dict):
                    continue
                ppx, ppy = p.get('x', 0), p.get('y', 0)
                pid = p.get('id', '?')
                ax.plot(ppx, ppy, 'D', color='#81C784', markersize=10,
                        markeredgecolor='black', markeredgewidth=0.8,
                        alpha=0.6, zorder=7,
                        label='Placard (expected)' if first_cfg else '')
                first_cfg = False
                yaw = p.get('yaw', 0)
                ddx = 0.8 * math.cos(yaw)
                ddy = 0.8 * math.sin(yaw)
                ax.annotate('', xy=(ppx + ddx, ppy + ddy), xytext=(ppx, ppy),
                            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))

        # Loop closure lines (if KISS-SLAM exited bootstrap)
        keyposes = self.get_keyposes()
        if self.closures and len(keyposes) > 0:
            first_lc = True
            for src_id, tgt_id in self.closures:
                if src_id < len(keyposes) and tgt_id < len(keyposes):
                    sx = keyposes[src_id][0, 3]
                    sy = keyposes[src_id][1, 3]
                    tx = keyposes[tgt_id][0, 3]
                    ty = keyposes[tgt_id][1, 3]
                    ax.plot([sx, tx], [sy, ty], '--', color='lime',
                            linewidth=1.5, alpha=0.6,
                            label='Loop closure' if first_lc else '')
                    first_lc = False

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Arena Map \u2014 UQ Space Theseus V \u2014 ARCh 2026',
                      fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.15, linestyle=':')

        # Scale bar
        sb_x = extent[0] + 0.5
        sb_y = extent[2] + 0.5
        ax.plot([sb_x, sb_x + 2], [sb_y, sb_y], 'k-', linewidth=4)
        ax.text(sb_x + 1, sb_y + 0.3, '2 m', ha='center', fontsize=10,
                fontweight='bold')

        # Stats box
        total_dist = 0.0
        if len(self._flio_path_xy) > 1:
            for i in range(1, len(self._flio_path_xy)):
                ddx = self._flio_path_xy[i][0] - self._flio_path_xy[i-1][0]
                ddy = self._flio_path_xy[i][1] - self._flio_path_xy[i-1][1]
                total_dist += math.hypot(ddx, ddy)

        stats_text = (
            f"Source: FLIO ikd-tree PCD\n"
            f"Points: {pcd.shape[0]:,}\n"
            f"Resolution: {res}m/cell\n"
            f"Travel: {total_dist:.1f}m\n"
            f"Cubes found: {len(self._cube_detections)}/4\n"
            f"Placards found: {len(self._aruco_detections)}/5\n"
            f"Loop closures: {len(self.closures)}")
        ax.text(extent[1] - 0.5, extent[2] + 0.5, stats_text,
                fontsize=9, fontfamily='monospace',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9))

        # Save (SIGINT-safe)
        path_out = os.path.join(self._save_dir, 'presentation_map.png')
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            fig.savefig(path_out, bbox_inches='tight', facecolor='white')
        finally:
            signal.signal(signal.SIGINT, old_handler)
        plt.close(fig)
        self.get_logger().info(
            f"Presentation map saved: {path_out} "
            f"({pcd.shape[0]:,} pts, {nx}x{ny} cells at {res}m)")

    @staticmethod
    def _read_pcd_numpy(path):
        """Read binary/ascii PCD file, return Nx3 float32 array."""
        try:
            with open(path, 'rb') as f:
                header = {}
                while True:
                    line = f.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('DATA binary_compressed'):
                        return None  # not supported
                    if line.startswith('DATA binary'):
                        break
                    if line.startswith('DATA ascii'):
                        n = int(header.get('POINTS', '0'))
                        pts = []
                        for _ in range(n):
                            parts = f.readline().decode('ascii').split()
                            if len(parts) >= 3:
                                pts.append([float(parts[0]),
                                            float(parts[1]),
                                            float(parts[2])])
                        return np.array(pts, dtype=np.float32) if pts else None
                    key, _, val = line.partition(' ')
                    header[key] = val

                n_points = int(header.get('POINTS', '0'))
                if n_points == 0:
                    return None

                fields = header.get('FIELDS', '').split()
                sizes = [int(s) for s in header.get('SIZE', '').split()]
                types = header.get('TYPE', '').split()
                if not fields or not sizes:
                    return None

                point_size = sum(sizes)
                raw = f.read(n_points * point_size)
                if len(raw) < n_points * point_size:
                    n_points = len(raw) // point_size

                offset = 0
                field_offsets = {}
                for i, name in enumerate(fields):
                    field_offsets[name] = (offset, sizes[i], types[i])
                    offset += sizes[i]

                data = np.frombuffer(raw, dtype=np.uint8).reshape(n_points, point_size)
                xyz = np.zeros((n_points, 3), dtype=np.float32)
                for j, axis in enumerate(['x', 'y', 'z']):
                    if axis in field_offsets:
                        off, sz, tp = field_offsets[axis]
                        col = data[:, off:off + sz].copy()
                        if tp == 'F' and sz == 4:
                            xyz[:, j] = col.view(np.float32).flatten()
                        elif tp == 'F' and sz == 8:
                            xyz[:, j] = col.view(np.float64).flatten().astype(np.float32)
                return xyz
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    # =====================================================================
    # Composite map generation (for presentation slides)
    # =====================================================================
    def _generate_composite_map(self):
        """Render annotated map PNG: occupancy + path + detections + arena."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            self.get_logger().warn("matplotlib not available — composite map skipped")
            return

        # Build point cloud (same logic as publish_2D_map)
        if self.voxel_maps:
            pcd = self._create_global_voxel_map()
            key_poses = self.get_keyposes()
            if len(key_poses) == 0:
                return
        elif (hasattr(self, 'odom_local_map')
              and self.odom_local_map.point_cloud().shape[0] > 100):
            pcd = self.odom_local_map.point_cloud()
            if hasattr(self, 'last_split_pose'):
                R_mat = self.last_split_pose[:3, :3]
                t_vec = self.last_split_pose[:3, 3]
                pcd = pcd @ R_mat.T + t_vec
        else:
            self.get_logger().warn("No point cloud data for composite map")
            return

        if pcd.shape[0] < 100:
            return

        res = self.config.occupancy_mapper.resolution

        # Compute grid extent from points
        xy_m = pcd[:, :2]
        pad_m = 2.0  # metres padding
        x_min = np.min(xy_m[:, 0]) - pad_m
        x_max = np.max(xy_m[:, 0]) + pad_m
        y_min = np.min(xy_m[:, 1]) - pad_m
        y_max = np.max(xy_m[:, 1]) + pad_m

        # Expand to include arena bounds if available
        cfg = self._arena_config
        if cfg and 'arena' in cfg:
            a = cfg['arena']
            x_min = min(x_min, a.get('x_min', x_min) - 1)
            x_max = max(x_max, a.get('x_max', x_max) + 1)
            y_min = min(y_min, a.get('y_min', y_min) - 1)
            y_max = max(y_max, a.get('y_max', y_max) + 1)

        # Bin points into 2D grid (max-Z per cell for heightmap colouring)
        nx = int((x_max - x_min) / res) + 1
        ny = int((y_max - y_min) / res) + 1
        cx = ((pcd[:, 0] - x_min) / res).astype(int)
        cy = ((pcd[:, 1] - y_min) / res).astype(int)
        valid = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        cx, cy = cx[valid], cy[valid]
        zvals = pcd[valid, 2]
        flat = cy * nx + cx
        n_cells = nx * ny

        # Occupancy: count points per cell
        counts = np.zeros(n_cells, dtype=np.int32)
        np.add.at(counts, flat, 1)
        max_z = np.full(n_cells, -np.inf, dtype=np.float64)
        np.maximum.at(max_z, flat, zvals)
        max_z[max_z == -np.inf] = np.nan

        # Build RGB image: grey=unknown, white=free, obstacle-coloured by height
        img = np.full((ny, nx, 3), 200, dtype=np.uint8)  # grey = unknown
        observed = counts > 0
        z_occ_min = self.config.occupancy_mapper.z_min
        z_occ_max = self.config.occupancy_mapper.z_max

        # Free cells: observed but below obstacle threshold
        obs_flat = np.where(observed)[0]
        for idx in obs_flat:
            z = max_z[idx]
            if np.isnan(z) or z < z_occ_min:
                r, c = divmod(idx, nx)
                img[r, c] = [240, 240, 240]  # light grey = free
            elif z <= z_occ_max:
                r, c = divmod(idx, nx)
                # Colour by height: blue(low) → red(high)
                t = np.clip((z - z_occ_min) / max(z_occ_max - z_occ_min, 0.01), 0, 1)
                img[r, c] = [int(t * 200), int((1 - abs(2 * t - 1)) * 150), int((1 - t) * 200)]

        img = np.flipud(img)
        extent = [x_min, x_max, y_min, y_max]

        # ── Plot ──
        fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi=150)
        ax.imshow(img, origin='lower', extent=extent, aspect='equal', alpha=0.85)

        # Legend proxy for obstacle cells (height-coloured in the occupancy image)
        from matplotlib.lines import Line2D
        ax.add_artist(Line2D([], [], marker='s', color='none', markerfacecolor='#4466AA',
                              markersize=10, label='Obstacles'))

        # Arena boundary
        if cfg and 'arena' in cfg:
            a = cfg['arena']
            bx = [a['x_min'], a['x_max']]
            by = [a['y_min'], a['y_max']]
            rect = Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                              linewidth=2.5, edgecolor='red', facecolor='none',
                              linestyle='--', label='Arena boundary')
            ax.add_patch(rect)

        # Robot path
        if len(self._robot_path_xy) > 1:
            px, py = zip(*self._robot_path_xy)
            ax.plot(px, py, '-', color='#2196F3', linewidth=2.0,
                    alpha=0.9, label='Robot path')
            ax.plot(px[0], py[0], '*', color='gold', markersize=18,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label='Start', zorder=10)
            ax.plot(px[-1], py[-1], 'o', color='#FF5722', markersize=10,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label='End', zorder=10)

        # Placards from config (known positions)
        if cfg and 'placards' in cfg:
            positions = cfg['placards']
            if isinstance(positions, dict):
                positions = positions.get('positions', {})
                if isinstance(positions, dict):
                    positions = positions.values()
            first = True
            for p in positions:
                if not isinstance(p, dict):
                    continue
                pid = p.get('id', '?')
                px, py = p.get('x', 0), p.get('y', 0)
                yaw = p.get('yaw', 0)
                ax.plot(px, py, 'D', color='#4CAF50', markersize=12,
                        markeredgecolor='black', markeredgewidth=0.8, zorder=8,
                        label='Placard (config)' if first else '')
                first = False
                ax.annotate(f'P{pid}', (px, py), textcoords='offset points',
                            xytext=(8, 8), fontsize=9, fontweight='bold',
                            color='#2E7D32',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
                # Facing direction arrow
                dx = 0.6 * math.cos(yaw)
                dy = 0.6 * math.sin(yaw)
                ax.annotate('', xy=(px + dx, py + dy), xytext=(px, py),
                            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))

        # ArUco detections (measured positions)
        first = True
        for aid, det in self._aruco_detections.items():
            ax.plot(det['x'], det['y'], '^', color='#FF9800', markersize=10,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=9,
                    label='ArUco detected' if first else '')
            first = False
            ax.annotate(f'A{aid}', (det['x'], det['y']),
                        textcoords='offset points', xytext=(8, -10),
                        fontsize=8, color='#E65100',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        # Cube detections
        first = True
        for cid, det in self._cube_detections.items():
            c = (det.get('r', 1), det.get('g', 0), det.get('b', 0))
            ax.plot(det['x'], det['y'], 's', color=c, markersize=12,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=9,
                    label='Cube detected' if first else '')
            first = False
            lbl = det.get('label', f'C{cid}')
            ax.annotate(lbl.upper(), (det['x'], det['y']),
                        textcoords='offset points', xytext=(8, 8),
                        fontsize=8, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        # SLAM keyposes
        keyposes = self.get_keyposes()
        if len(keyposes) > 1:
            kx = [kp[0, 3] for kp in keyposes]
            ky = [kp[1, 3] for kp in keyposes]
            ax.plot(kx, ky, '.', color='cyan', markersize=4, alpha=0.6,
                    label='SLAM keyposes')

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title('KISS-SLAM Composite Map — UQ Space ARCh 2026', fontsize=14,
                      fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle=':')

        # Scale bar
        sb_x = extent[0] + 0.5
        sb_y = extent[2] + 0.5
        ax.plot([sb_x, sb_x + 2], [sb_y, sb_y], 'k-', linewidth=3)
        ax.text(sb_x + 1, sb_y + 0.2, '2 m', ha='center', fontsize=9,
                fontweight='bold')

        path_out = os.path.join(self._save_dir, 'composite_map.png')
        fig.savefig(path_out, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.get_logger().info(f"Composite map saved: {path_out}")

    # =====================================================================
    # Mission report generation
    # =====================================================================
    def _load_corrected_cubes(self):
        """Load drift-corrected cube coordinates if available."""
        # Check run dir first, then /tmp fallback
        candidates = [
            os.path.join(self._save_dir, 'cube_corrected_coordinates.json'),
            '/tmp/cube_corrected_coordinates.json',
        ]
        for corrected_path in candidates:
            if os.path.exists(corrected_path):
                try:
                    with open(corrected_path, 'r') as f:
                        corrected = json.load(f)
                    self.get_logger().info(
                        f'Loaded {len(corrected)} corrected cube coordinates from {corrected_path}')
                    return corrected
                except Exception as e:
                    self.get_logger().warn(f'Failed to load corrected cubes from {corrected_path}: {e}')
        return None

    def _generate_mission_report(self):
        """Save structured mission report JSON."""
        duration = time.time() - self._start_time
        keyposes = self.get_keyposes()

        # Arena info
        arena_info = {}
        placard_info = []
        cfg = self._arena_config
        if cfg:
            if 'arena' in cfg:
                arena_info = cfg['arena']
            if 'placards' in cfg:
                positions = cfg['placards']
                if isinstance(positions, dict):
                    positions = positions.get('positions', {})
                    if isinstance(positions, dict):
                        positions = positions.values()
                for p in positions:
                    if not isinstance(p, dict):
                        continue
                    pid = p.get('id', -1)
                    det = self._aruco_detections.get(pid)
                    placard_info.append({
                        'id': pid,
                        'known_position': {'x': p.get('x'), 'y': p.get('y')},
                        'marker_height_m': p.get('z', 0),
                        'yaw_rad': p.get('yaw', 0),
                        'detected': det is not None,
                        'detected_position': det if det else None,
                    })

        # Cube detections
        cube_info = []
        for cid, det in self._cube_detections.items():
            cube_info.append({
                'id': cid,
                'label': det.get('label', ''),
                'position': {'x': round(det['x'], 3), 'y': round(det['y'], 3),
                              'z': round(det['z'], 3)},
            })

        report = {
            'run_timestamp': os.path.basename(self._save_dir),
            'duration_s': round(duration, 1),
            'distance_m': round(self._total_travel_m, 2),
            'slam': {
                'local_maps': len(self.voxel_maps),
                'loop_closures': len(self.closures),
                'closure_pairs': self.closures,
                'last_correction_m': round(self._last_correction_norm, 4),
                'keyposes_count': len(keyposes),
            },
            'arena': arena_info,
            'placards': placard_info,
            'cubes': cube_info,
            'cubes_corrected': self._load_corrected_cubes(),
            'path_samples': len(self._robot_path_xy),
            'flio_trajectory_samples': len(self._flio_path_xy),
            'outputs': [f for f in os.listdir(self._save_dir)
                        if os.path.isfile(os.path.join(self._save_dir, f))],
        }

        report_path = os.path.join(self._save_dir, 'mission_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        self.get_logger().info(
            f"Mission report saved: {report_path} "
            f"(duration={duration:.0f}s, dist={self._total_travel_m:.1f}m, "
            f"placards={sum(1 for p in placard_info if p['detected'])}/{len(placard_info)}, "
            f"cubes={len(cube_info)})"
        )

    # =====================================================================
    # FAST-LIO equivalent: RGBpointBodyToWorld
    #
    # C++: V3D p_global(state_point.rot *
    #        (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I)
    #        + state_point.pos);
    #
    # No IMU in KISS → no LiDAR-to-IMU extrinsic (offset_R_L_I, offset_T_L_I).
    # Simplifies to: p_global = R_world * p_body + t_world
    # =====================================================================
    @staticmethod
    def _points_body_to_world(body_points, pose):
        """Transform body-frame points to world frame. FAST-LIO: RGBpointBodyToWorld."""
        return body_points @ pose[:3, :3].T + pose[:3, 3]

    # =====================================================================
    # FAST-LIO equivalent: publish_frame_world
    #
    # C++ (laserMapping.cpp:478-529):
    #   void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
    #   {
    #       if(scan_pub_en)
    #       {
    #           PointCloudXYZI::Ptr laserCloudFullRes(
    #               dense_pub_en ? feats_undistort : feats_down_body);
    #           // transform body → world
    #           for (int i = 0; i < size; i++)
    #               RGBpointBodyToWorld(&laserCloudFullRes->points[i],
    #                                   &laserCloudWorld->points[i]);
    #           // publish
    #           pubLaserCloudFull.publish(laserCloudmsg);
    #       }
    #
    #       /*** save map ***/
    #       if (pcd_save_en)
    #       {
    #           // always uses feats_undistort (full res) for saving
    #           for (int i = 0; i < size; i++)
    #               RGBpointBodyToWorld(&feats_undistort->points[i],
    #                                   &laserCloudWorld->points[i]);
    #           *pcl_wait_save += *laserCloudWorld;
    #
    #           scan_wait_num++;
    #           if (pcl_wait_save->size() > 0
    #               && pcd_save_interval > 0
    #               && scan_wait_num >= pcd_save_interval)
    #           {
    #               pcd_writer.writeBinary(filename, *pcl_wait_save);
    #               pcl_wait_save->clear();
    #               scan_wait_num = 0;
    #           }
    #       }
    #   }
    # =====================================================================
    def publish_frame_world(self, body_points, current_pose, stamp):
        """FAST-LIO: publish_frame_world — per-scan world-frame cloud + PCD accumulation."""

        # --- Publish per-scan world-frame cloud ---
        # C++: if(scan_pub_en) { ... pubLaserCloudFull.publish(...); }
        if self.scan_pub_en:
            # C++: dense_pub_en ? feats_undistort : feats_down_body
            if self.dense_pub_en:
                pub_points = body_points
            else:
                pub_points = voxel_down_sample(body_points, self.local_map_voxel_size)

            world_points_pub = self._points_body_to_world(pub_points, current_pose)

            header = std_msgs.msg.Header()
            header.stamp = stamp
            header.frame_id = self.map_frame    # FAST-LIO: "camera_init"
            cloud_msg = pc2.create_cloud_xyz32(header, world_points_pub[:, :3])
            self.cloud_registered_pub.publish(cloud_msg)

    # =====================================================================
    # Main keyframe callback
    #
    # FAST-LIO main loop equivalent (laserMapping.cpp:860-984):
    #   1. Process scan
    #   2. Publish odometry
    #   3. map_incremental()
    #   4. if (scan_pub_en || pcd_save_en) publish_frame_world(...)
    # =====================================================================
    def keyframe_callback(self, deskewed_points_msg: PointCloud2, odom_msg: Odometry):
        stamp = odom_msg.header.stamp

        # 1. Extract data
        points_np = pc2.read_points(deskewed_points_msg, field_names=("x", "y", "z"), skip_nans=True)
        keyframe_points = np.vstack([points_np['x'], points_np['y'], points_np['z']]).T.astype(np.float64)
        current_keyframe_pose = self._msg_to_pose(odom_msg.pose.pose)
        relative_motion = np.linalg.inv(self.last_split_pose) @ current_keyframe_pose
        self._current_odom_pose = np.copy(current_keyframe_pose)

        # C2: Travel distance
        if self._last_pose_for_travel is not None:
            delta = np.linalg.norm(current_keyframe_pose[:3, 3] - self._last_pose_for_travel[:3, 3])
            self._total_travel_m += delta
        self._last_pose_for_travel = np.copy(current_keyframe_pose)

        # 2. Update voxel grid and local map graph
        mapping_frame = voxel_down_sample(keyframe_points, self.local_map_voxel_size)
        self.voxel_grid.integrate_frame(mapping_frame, relative_motion)
        self.odom_local_map.integrate_frame(keyframe_points, relative_motion)
        self.local_map_graph.last_local_map.local_trajectory.append(relative_motion)

        # 3. map→odom TF — use cached _map_T_odom instead of rebuilding full pose history
        corrected_pose = self._map_T_odom @ self._current_odom_pose
        self._publish_transform(self._map_T_odom, stamp, self.map_frame, self.odom_frame)
        self._publish_pose(corrected_pose, stamp, self.map_frame)

        # 4. FAST-LIO equivalent: publish_frame_world
        if self.scan_pub_en:
            self.publish_frame_world(keyframe_points, corrected_pose, stamp)

        # 5. Check split
        current_step_distance = np.linalg.norm(relative_motion[:3, -1])
        if current_step_distance > self.local_map_splitting_distance:
            self._split_local_map(current_keyframe_pose, stamp)

    # =====================================================================
    # Local map split
    # =====================================================================
    def _split_local_map(self, current_keyframe_pose, stamp):
        self.last_split_pose = np.copy(current_keyframe_pose)
        last_local_map = self.local_map_graph.last_local_map
        last_pose_in_local = last_local_map.local_trajectory[-1]

        # A2: Store numpy positions, not Open3D PCDs
        o3d_pcd = self.voxel_grid.open3d_pcd_with_normals()
        positions_np = o3d_pcd.point.positions.numpy().astype(np.float64)
        self.voxel_maps.append(positions_np)

        transformed_local_map = self._transform_points(
            self.odom_local_map.point_cloud(),
            np.linalg.inv(last_local_map.local_trajectory[-1])
        )
        self.odom_local_map.clear()

        query_id = last_local_map.id
        query_points = self.voxel_grid.point_cloud()
        self.local_map_graph.finalize_local_map(self.voxel_grid)
        self.voxel_grid.clear()
        self.voxel_grid.add_points(transformed_local_map)

        self.optimizer.add_variable(
            self.local_map_graph.last_id,
            self.local_map_graph.last_keypose
        )
        # B1: Odometry factor — unit weight
        self.optimizer.add_factor(
            self.local_map_graph.last_id, query_id,
            last_pose_in_local, _ODOM_INFORMATION
        )
        self._compute_closures(query_id, query_points)

        # Refresh map→odom after split (poses changed with new keypose)
        poses = self.poses
        if len(poses) > 0:
            self._map_T_odom = poses[-1] @ np.linalg.inv(self._current_odom_pose)

        gmap = self._get_cached_global_map()
        self.publish_pc2(gmap, frame_id=self.map_frame, stamp=stamp)

    # =====================================================================
    # A1: Cached global map
    # =====================================================================
    def _get_cached_global_map(self):
        n_maps = len(self.voxel_maps)
        if n_maps == 0:
            return np.empty((0, 3), dtype=np.float64)

        keyposes = self.get_keyposes()

        if self._cache_dirty:
            all_points = []
            for i in range(min(n_maps, len(keyposes))):
                pts = self.voxel_maps[i]
                if pts.shape[0] == 0:
                    continue
                pose = keyposes[i]
                all_points.append(pts @ pose[:3, :3].T + pose[:3, 3])
            if all_points:
                self._cached_global_map = voxel_down_sample(
                    np.vstack(all_points), self.local_map_voxel_size
                )
            else:
                self._cached_global_map = np.empty((0, 3), dtype=np.float64)
            self._cached_map_count = n_maps
            self._cache_dirty = False

        elif n_maps > self._cached_map_count:
            new_points = []
            for i in range(self._cached_map_count, min(n_maps, len(keyposes))):
                pts = self.voxel_maps[i]
                if pts.shape[0] == 0:
                    continue
                pose = keyposes[i]
                new_points.append(pts @ pose[:3, :3].T + pose[:3, 3])
            if new_points:
                new_down = voxel_down_sample(np.vstack(new_points), self.local_map_voxel_size)
                if self._cached_global_map.shape[0] > 0:
                    self._cached_global_map = np.vstack([self._cached_global_map, new_down])
                else:
                    self._cached_global_map = new_down
            self._cached_map_count = n_maps

        return self._cached_global_map

    def _create_global_voxel_map(self):
        return self._get_cached_global_map()

    # =====================================================================
    # Loop closure + PGO
    # =====================================================================
    def _compute_closures(self, query_id, query):
        is_good, source_id, target_id, pose_constraint = self.closer.compute(
            query_id, query, self.local_map_graph
        )
        if is_good:
            self.closures.append((source_id, target_id))
            # B1: Closure factor — tighter weight
            self.optimizer.add_factor(
                source_id, target_id, pose_constraint, _CLOSURE_INFORMATION
            )
            self._optimize_pose_graph()
            self.get_logger().info(
                f"Loop closure accepted: {source_id} -> {target_id}. "
                f"Total closures: {len(self.closures)}"
            )

    def _optimize_pose_graph(self):
        self.optimizer.optimize()
        estimates = self.optimizer.estimates()
        max_correction = 0.0
        for id_, pose in estimates.items():
            old_pose = self.local_map_graph[id_].keypose
            correction = np.linalg.norm(pose[:3, 3] - old_pose[:3, 3])
            max_correction = max(max_correction, correction)
            self.local_map_graph[id_].keypose = np.copy(pose)
        self._last_correction_norm = max_correction
        # A1: Invalidate cache
        self._cache_dirty = True
        # Force occupancy grid recomputation after loop closure
        self._last_map_hash = None
        # Update map→odom correction from optimised poses
        poses = self.poses
        if len(poses) > 0:
            self._map_T_odom = poses[-1] @ np.linalg.inv(self._current_odom_pose)

    # =====================================================================
    # Message conversion + publishers
    # =====================================================================
    def _msg_to_pose(self, pose_msg) -> np.ndarray:
        p = pose_msg.position
        o = pose_msg.orientation
        pose = np.eye(4)
        pose[:3, 3] = [p.x, p.y, p.z]
        pose[:3, :3] = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        return pose

    def _transform_points(self, pcd, T):
        rot = T[:3, :3]
        t = T[:3, -1]
        return pcd @ rot.T + t

    def _publish_pose(self, pose: np.ndarray, stamp, frame_id: str):
        p = PoseStamped()
        p.header.stamp = stamp
        p.header.frame_id = frame_id
        p.pose.position.x = pose[0, 3]
        p.pose.position.y = pose[1, 3]
        p.pose.position.z = pose[2, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]
        self.pose_pub.publish(p)

    def _publish_transform(self, pose: np.ndarray, stamp, frame_id: str, child_frame_id: str):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, frame_id, child_frame_id
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pose[0, 3], pose[1, 3], pose[2, 3]
        q = R.from_matrix(pose[:3, :3]).as_quat()
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q[0], q[1], q[2], q[3]
        self.tf_broadcaster.sendTransform(t)

    def publish_pc2(self, points, frame_id=None, stamp=None):
        if hasattr(points, 'point'):
            points_array = points.point.positions.cpu().numpy()
        elif hasattr(points, 'points'):
            points_array = np.asarray(points.points)
        else:
            points_array = points
        if points_array.shape[0] == 0:
            return
        header = std_msgs.msg.Header()
        header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        header.frame_id = frame_id if frame_id is not None else self.map_frame
        msg = pc2.create_cloud_xyz32(header, points_array[:, :3])
        self.global_voxel_map_pub.publish(msg)

    # =====================================================================
    # B2: 2D Occupancy Grid with stable origin
    # =====================================================================
    @timing_decorator
    def publish_2D_map(self):
        """Generate 2D occupancy grid from 3D voxel data for Nav2 global planner."""
        if self.voxel_maps:
            pcd = self._create_global_voxel_map()
            key_poses = self.get_keyposes()
            if len(key_poses) == 0:
                return
            origin_pose = key_poses[0]
        elif (hasattr(self, 'odom_local_map')
              and self.odom_local_map.point_cloud().shape[0] > 100):
            pcd = self.odom_local_map.point_cloud()
            if hasattr(self, 'last_split_pose'):
                origin_pose = self.last_split_pose
                R_mat = origin_pose[:3, :3]
                t_vec = origin_pose[:3, 3]
                pcd = pcd @ R_mat.T + t_vec
            else:
                origin_pose = np.eye(4)
            self.get_logger().info(
                f'[Bootstrap] Publishing map from odom_local_map '
                f'({pcd.shape[0]} points)',
                throttle_duration_sec=5.0,
            )
        else:
            return

        if pcd.shape[0] < 100:
            return

        # Clip points to arena bounds (avoid processing off-arena LiDAR returns)
        if self._arena_config and 'arena' in self._arena_config:
            a = self._arena_config['arena']
            pad = 2.0  # small pad for map edge rendering
            arena_mask = (
                (pcd[:, 0] >= a['x_min'] - pad) & (pcd[:, 0] <= a['x_max'] + pad) &
                (pcd[:, 1] >= a['y_min'] - pad) & (pcd[:, 1] <= a['y_max'] + pad)
            )
            pcd = pcd[arena_mask]
            if pcd.shape[0] < 100:
                return

        # Hash gating: skip rebuild if point cloud data unchanged (~5ms)
        pcd_hash = hashlib.md5(pcd.tobytes()).digest()
        if pcd_hash == self._last_map_hash and self._last_occupancy_msg is not None:
            self._last_occupancy_msg.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self._last_occupancy_msg)
            return

        self._last_map_hash = pcd_hash
        occupancy_mapper = OccupancyGridMapper(self.config.occupancy_mapper)
        occupancy_mapper.integrate_frame(pcd, origin_pose)
        occupancy_mapper.compute_3d_occupancy_information()

        active_voxels = occupancy_mapper.active_voxels
        occ_values = occupancy_mapper.occupancies
        if len(active_voxels) == 0:
            return

        res = self.config.occupancy_mapper.resolution
        min_z_idx = int(self.config.occupancy_mapper.z_min // res)
        max_z_idx = int(self.config.occupancy_mapper.z_max // res)
        z_mask = ((active_voxels[:, 2] >= min_z_idx)
                  & (active_voxels[:, 2] <= max_z_idx))
        if not np.any(z_mask):
            return

        voxels = active_voxels[z_mask]
        occs = occ_values[z_mask]

        # B2: Stable origin — clamp grid to arena bounds when available
        xy = voxels[:, :2]

        if self._arena_config and 'arena' in self._arena_config:
            # Fixed grid covering exactly the arena, with a 1-cell wall band
            a = self._arena_config['arena']
            WALL_CELLS = int(math.ceil(1.0 / res))  # 1m wall band in cells
            arena_x_min_vox = int(math.floor(a['x_min'] / res)) - WALL_CELLS
            arena_x_max_vox = int(math.ceil(a['x_max'] / res)) + WALL_CELLS
            arena_y_min_vox = int(math.floor(a['y_min'] / res)) - WALL_CELLS
            arena_y_max_vox = int(math.ceil(a['y_max'] / res)) + WALL_CELLS
            self._og_origin = np.array([arena_x_min_vox, arena_y_min_vox])
            nx = arena_x_max_vox - arena_x_min_vox
            ny = arena_y_max_vox - arena_y_min_vox
            self._og_size = (nx, ny)
        else:
            # No arena config — grow dynamically (original behaviour)
            data_lower = np.min(xy, axis=0)
            data_upper = np.max(xy, axis=0)
            PAD = 20
            if self._og_origin is None:
                self._og_origin = data_lower - PAD
                self._og_size = tuple((data_upper - self._og_origin + 1 + PAD).astype(int))
            else:
                new_lower = np.minimum(self._og_origin, data_lower - PAD)
                old_upper = self._og_origin + np.array(self._og_size)
                new_upper = np.maximum(old_upper, data_upper + 1 + PAD)
                self._og_origin = new_lower
                self._og_size = tuple((new_upper - new_lower).astype(int))
            nx, ny = self._og_size

        n_cells = int(nx) * int(ny)
        x_rel = (xy[:, 0] - self._og_origin[0]).astype(int)
        y_rel = (xy[:, 1] - self._og_origin[1]).astype(int)
        flat_idx = y_rel * nx + x_rel
        valid = (flat_idx >= 0) & (flat_idx < n_cells)
        if not np.all(valid):
            flat_idx = flat_idx[valid]
            occs = occs[valid]

        max_occ = np.full(n_cells, -np.inf, dtype=np.float64)
        np.maximum.at(max_occ, flat_idx, occs)

        free_thresh = self.config.occupancy_mapper.free_threshold
        occ_thresh = self.config.occupancy_mapper.occupied_threshold
        grid = np.full(n_cells, -1, dtype=np.int8)
        observed = np.isfinite(max_occ)
        grid[observed & (max_occ < free_thresh)] = 0
        grid[observed & (max_occ > occ_thresh)] = 100

        # Arena boundary wall: mark cells outside nav_bounds as occupied (100)
        if self._arena_config and 'arena' in self._arena_config:
            a = self._arena_config['arena']
            nav_x_min_vox = int(math.floor(a['x_min'] / res))
            nav_x_max_vox = int(math.ceil(a['x_max'] / res))
            nav_y_min_vox = int(math.floor(a['y_min'] / res))
            nav_y_max_vox = int(math.ceil(a['y_max'] / res))
            grid_2d = grid.reshape((ny, nx))
            gx_vox = self._og_origin[0] + np.arange(nx)
            gy_vox = self._og_origin[1] + np.arange(ny)
            outside_x = (gx_vox < nav_x_min_vox) | (gx_vox >= nav_x_max_vox)
            outside_y = (gy_vox < nav_y_min_vox) | (gy_vox >= nav_y_max_vox)
            # Rows entirely outside Y bounds
            grid_2d[outside_y, :] = 100
            # Columns entirely outside X bounds
            grid_2d[:, outside_x] = 100
            grid = grid_2d.ravel()

        n_free = int(np.sum(grid == 0))
        n_occ = int(np.sum(grid == 100))
        n_unk = int(np.sum(grid == -1))
        self.get_logger().info(
            f'Map: {nx}x{ny}, free={n_free}, occ={n_occ}, unknown={n_unk}',
            throttle_duration_sec=5.0,
        )

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = res
        msg.info.width = int(nx)
        msg.info.height = int(ny)
        msg.info.origin.position.x = float(self._og_origin[0]) * res
        msg.info.origin.position.y = float(self._og_origin[1]) * res
        msg.data = grid.tolist()
        self._last_occupancy_msg = msg
        self.map_pub.publish(msg)

    # =====================================================================
    # C2: SLAM status
    # =====================================================================
    def _publish_status(self):
        status = {
            'local_maps': len(self.voxel_maps),
            'loop_closures': len(self.closures),
            'travel_m': round(self._total_travel_m, 2),
            'last_correction_m': round(self._last_correction_norm, 4),
            'splitting_distance_m': self.local_map_splitting_distance,
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    # =====================================================================
    # C1: Map save
    # =====================================================================
    def _save_map_callback(self, request, response):
        try:
            self._save_maps_to_disk()
            response.success = True
            response.message = f"Maps saved to {self._save_dir}"
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f"Map save failed: {e}")
        return response

    def _save_maps_to_disk(self):
        os.makedirs(self._save_dir, exist_ok=True)

        # Symlink <base_dir>/latest → this run's folder
        latest = os.path.join(os.path.dirname(self._save_dir), 'latest')
        try:
            if os.path.islink(latest):
                os.remove(latest)
            os.symlink(self._save_dir, latest)
        except OSError:
            pass

        # 1. Save status (instant)
        status_path = os.path.join(self._save_dir, 'slam_status.json')
        status = {
            'local_maps': len(self.voxel_maps),
            'loop_closures': len(self.closures),
            'closure_pairs': self.closures,
            'travel_m': round(self._total_travel_m, 2),
            'last_correction_m': round(self._last_correction_norm, 4),
            'keyposes': [kp.tolist() for kp in self.get_keyposes()],
        }
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        self.get_logger().info(f"Status saved: {status_path}")

        # 2. Save nav2-compatible .pgm/.yaml map (instant, from cached grid)
        try:
            self._save_nav2_map()
        except Exception as e:
            import traceback
            self.get_logger().error(f"Nav2 map save failed: {e}\n{traceback.format_exc()}")

        # 3. Composite annotated map (fast, uses in-memory data)
        try:
            self._generate_composite_map()
        except Exception as e:
            import traceback
            self.get_logger().error(f"Composite map failed: {e}\n{traceback.format_exc()}")

        # 4. Mission report (instant)
        try:
            self._generate_mission_report()
        except Exception as e:
            import traceback
            self.get_logger().error(f"Mission report failed: {e}\n{traceback.format_exc()}")

        # 5. Copy FLIO PCD into session directory (SLOW — waits up to 10s)
        #    FAST-LIO saves on SIGINT via its destructor, which races with this
        #    shutdown handler.  Wait for the file mtime to be AFTER node start
        #    so we don't copy a stale PCD from a previous run.
        import shutil
        import time as _time
        flio_pcd_src = os.path.expanduser(
            '~/Documents/theseus_autonomy_ws/data/maps/hardware/flio_map.pcd')
        flio_pcd_dst = os.path.join(self._save_dir, 'flio_map.pcd')
        for attempt in range(20):  # up to ~10s
            if os.path.exists(flio_pcd_src):
                try:
                    src_mtime = os.path.getmtime(flio_pcd_src)
                    if src_mtime < self._start_time:
                        self.get_logger().info(
                            f"FLIO PCD stale (mtime {src_mtime:.0f} < start {self._start_time:.0f}), "
                            f"waiting... ({attempt + 1}/20)")
                        _time.sleep(0.5)
                        continue
                    src_size = os.path.getsize(flio_pcd_src)
                    _time.sleep(0.5)
                    if os.path.getsize(flio_pcd_src) == src_size:
                        shutil.copy2(flio_pcd_src, flio_pcd_dst)
                        self.get_logger().info(
                            f"FLIO PCD ({src_size / 1e6:.1f} MB) → {flio_pcd_dst}")
                        break
                except Exception as e:
                    self.get_logger().warn(f"FLIO PCD copy attempt {attempt}: {e}")
            _time.sleep(0.5)
        else:
            self.get_logger().warn(
                f"FLIO PCD not found or still stale at {flio_pcd_src} after 10s")

        # 6. Presentation map from FLIO PCD (needs PCD from step 5)
        try:
            self._generate_presentation_map()
        except Exception as e:
            import traceback
            self.get_logger().error(f"Presentation map failed: {e}\n{traceback.format_exc()}")

        # 7. Pull ArUco captures from Jetson into session folder
        self._pull_jetson_captures()

        # 8. Rsync session folder to laptop (blocks until complete)
        if self._sync_destinations:
            self._sync_to_laptop()

    def _save_nav2_map(self):
        """Save the last occupancy grid as a nav2-compatible .pgm + .yaml pair."""
        msg = self._last_occupancy_msg
        if msg is None:
            self.get_logger().warn("No occupancy grid to save — skipping nav2 map")
            return

        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Convert OccupancyGrid data to PGM pixels
        # OccupancyGrid: -1=unknown, 0=free, 100=occupied
        # PGM (nav2 convention): 254=free, 0=occupied, 205=unknown
        import struct
        pixels = bytearray(width * height)
        for i, val in enumerate(msg.data):
            if val < 0:
                pixels[i] = 205  # unknown
            else:
                pixels[i] = max(0, min(254, 254 - int(val * 254 / 100)))

        # PGM is stored top-row-first, OccupancyGrid is bottom-row-first
        flipped = bytearray(width * height)
        for row in range(height):
            src_start = row * width
            dst_start = (height - 1 - row) * width
            flipped[dst_start:dst_start + width] = pixels[src_start:src_start + width]

        pgm_path = os.path.join(self._save_dir, 'map.pgm')
        yaml_path = os.path.join(self._save_dir, 'map.yaml')

        with open(pgm_path, 'wb') as f:
            header = f"P5\n{width} {height}\n255\n"
            f.write(header.encode('ascii'))
            f.write(flipped)

        with open(yaml_path, 'w') as f:
            f.write(f"image: map.pgm\n")
            f.write(f"resolution: {res}\n")
            f.write(f"origin: [{origin_x}, {origin_y}, 0.0]\n")
            f.write(f"negate: 0\n")
            f.write(f"occupied_thresh: 0.65\n")
            f.write(f"free_thresh: 0.196\n")

        pgm_size = os.path.getsize(pgm_path)
        self.get_logger().info(
            f"Nav2 map saved: {pgm_path} ({width}x{height}, {res}m/px, {pgm_size/1e3:.0f} KB)")

    def _pull_jetson_captures(self):
        """Pull ArUco snapshots from Jetson into the session folder."""
        import subprocess
        jetson_src = 'uqs@192.168.10.3:/home/uqs/ros2_ws/aruco_captures/'
        local_dst = os.path.join(self._save_dir, 'aruco_captures/')
        try:
            result = subprocess.run(
                ['rsync', '-az',
                 '-e', 'ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no',
                 jetson_src, local_dst],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                n = len(os.listdir(local_dst)) if os.path.isdir(local_dst) else 0
                self.get_logger().info(f"Pulled {n} ArUco captures from Jetson")
            else:
                self.get_logger().warn(f"Jetson capture pull failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Jetson capture pull timed out (3s)")
        except FileNotFoundError:
            self.get_logger().warn("rsync not available — skipping Jetson capture pull")
        except Exception as e:
            self.get_logger().warn(f"Jetson capture pull error: {e}")

    def _sync_to_laptop(self):
        """Rsync session folder to laptop. Tries each destination until one works."""
        import subprocess
        import signal

        src = self._save_dir.rstrip('/') + '/'
        session_name = os.path.basename(self._save_dir)

        # Mask SIGINT so a second Ctrl+C doesn't kill the transfer
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for dest in self._sync_destinations:
                remote_dir = dest.rstrip('/') + '/' + session_name + '/'
                self.get_logger().info(f"Syncing session to: {remote_dir}")
                try:
                    result = subprocess.run(
                        ['rsync', '-az', '--info=progress2',
                         '-e', 'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no',
                         src, remote_dir],
                        capture_output=True, text=True, timeout=120,
                    )
                    if result.returncode == 0:
                        self.get_logger().info(f"Session synced: {remote_dir}")
                        return  # success — done
                    else:
                        self.get_logger().warn(
                            f"Rsync failed for {dest} (rc={result.returncode}): "
                            f"{result.stderr.strip()}")
                except subprocess.TimeoutExpired:
                    self.get_logger().warn(f"Rsync timed out for {dest}")
                except FileNotFoundError:
                    self.get_logger().warn("rsync not installed on OPi")
                    return
                except Exception as e:
                    self.get_logger().warn(f"Rsync error for {dest}: {e}")

            self.get_logger().warn(
                f"All sync destinations failed. Manual sync:\n"
                f"  rsync -az {src} <laptop>:~/theseus_autonomy_ws/data/maps/hardware/{session_name}/")
        finally:
            signal.signal(signal.SIGINT, old_handler)

    # =====================================================================
    # Utility
    # =====================================================================
    @property
    def poses(self):
        poses = [np.eye(4)]
        for node in self.local_map_graph.local_maps():
            for rel_pose in node.local_trajectory[1:]:
                poses.append(node.keypose @ rel_pose)
        return poses

    def get_keyposes(self):
        return list(self.local_map_graph.keyposes())

    def _fine_grained_optimization(self):
        pgo = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        id_ = 0
        pgo.add_variable(id_, self.local_map_graph[id_].keypose)
        pgo.fix_variable(id_)
        for node in self.local_map_graph.local_maps():
            odometry_factors = [
                np.linalg.inv(T0) @ T1
                for T0, T1 in zip(node.local_trajectory[:-1], node.local_trajectory[1:])
            ]
            for i, factor in enumerate(odometry_factors):
                pgo.add_variable(id_ + 1, node.keypose @ node.local_trajectory[i + 1])
                pgo.add_factor(id_ + 1, id_, factor, np.eye(6))
                id_ += 1
            pgo.fix_variable(id_ - 1)
        pgo.optimize()
        poses = [x for x in pgo.estimates().values()]
        return poses, pgo

    # =====================================================================
    # Shutdown
    # =====================================================================
    def destroy_node(self):
        self.get_logger().info("Shutting down — saving maps...")
        try:
            self._save_maps_to_disk()
        except Exception as e:
            self.get_logger().error(f"Failed to save maps on shutdown: {e}")
        self.get_logger().info("SLAM node shutdown complete.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
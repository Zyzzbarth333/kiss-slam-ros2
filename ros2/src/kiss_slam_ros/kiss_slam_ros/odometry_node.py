import rclpy
import numpy as np
import std_msgs.msg

from kiss_icp.kiss_icp import KissICP

from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2 as pc2

import tf2_ros
from kiss_slam_ros.utils.config import declare_parameters, get_kiss_icp_config

# Timestamp field names to search for, in priority order.
# LIVOX MID-360 PointXYZRTLT uses 'timestamp' (float64).
# Ouster / Velodyne / Gazebo typically use 't' (float32/float64).
# Some drivers use 'time' or 'time_stamp'.
_TIMESTAMP_FIELDS = ('t', 'timestamp', 'time', 'time_stamp')


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        # Declare and get parameters
        declare_parameters(self)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        config = get_kiss_icp_config(self)
        self.get_logger().info("KISS-ICP ROS2 node configured successfully.")

        # Core Odometry component
        self.odometry = KissICP(config)

        # Detected timestamp field name (resolved on first message)
        self._ts_field = None
        self._ts_field_resolved = False

        # ROS Communications
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._init_publishers()
        self._init_subscribers()

        self.get_logger().info(
            f"Odometry Node initialized with odom_frame: {self.odom_frame}, base_frame: {self.base_frame}"
        )
        self.get_logger().info("Odometry Node initialized. Waiting for data...")

    def _init_publishers(self):
        """Initialize all ROS publishers."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.deskewed_pub = self.create_publisher(PointCloud2, 'deskewed_points', qos)
        self.odom_pub = self.create_publisher(Odometry, 'odometry', qos)

    def _init_subscribers(self):
        """Initialize subscribers."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PointCloud2, '/points_raw', self.listener_callback, qos
        )

    def _resolve_timestamp_field(self, msg_fields: dict):
        """Detect the per-point timestamp field on the first message."""
        for candidate in _TIMESTAMP_FIELDS:
            if candidate in msg_fields:
                self._ts_field = candidate
                self.get_logger().info(
                    f"Deskew: using per-point timestamp field '{candidate}'"
                )
                break
        if self._ts_field is None:
            self.get_logger().warn(
                f"No per-point timestamp field found (checked {_TIMESTAMP_FIELDS}). "
                f"Available fields: {list(msg_fields.keys())}. "
                f"Deskewing will be disabled."
            )
        self._ts_field_resolved = True

    def listener_callback(self, msg: PointCloud2):
        """Main callback for processing point cloud data and publishing odometry."""
        msg_fields = {field.name: field for field in msg.fields}

        # Resolve timestamp field once
        if not self._ts_field_resolved:
            self._resolve_timestamp_field(msg_fields)

        # Extract points + timestamps
        if self._ts_field is not None:
            points_np = pc2.read_points(
                msg, field_names=("x", "y", "z", self._ts_field), skip_nans=True
            )
            timestamps = points_np[self._ts_field].astype(np.float64)
        else:
            points_np = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
            timestamps = np.array([])

        points = np.vstack([points_np['x'], points_np['y'], points_np['z']]).T.astype(np.float64)

        # Run KISS-ICP odometry
        deskewed_frame, _ = self.odometry.register_frame(points, timestamps)
        current_odom_pose = self.odometry.last_pose

        # Publish the odom -> base_link transform and odometry
        self._publish_transform(current_odom_pose, msg.header.stamp, self.odom_frame, self.base_frame)
        self._publish_odometry(current_odom_pose, msg.header.stamp, self.odom_frame, self.base_frame)
        self._publish_deskewed_points(deskewed_frame, msg.header.stamp)

    def _publish_transform(self, pose: np.ndarray, stamp, frame_id: str, child_frame_id: str):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = pose[0, 3]
        t.transform.translation.y = pose[1, 3]
        t.transform.translation.z = pose[2, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

    def _publish_odometry(self, pose: np.ndarray, stamp, frame_id: str, child_frame_id: str):
        """Publish odometry message."""
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = frame_id
        odom.child_frame_id = child_frame_id

        odom.pose.pose.position.x = pose[0, 3]
        odom.pose.pose.position.y = pose[1, 3]
        odom.pose.pose.position.z = pose[2, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        odom.pose.covariance[0] = 0.1   # x
        odom.pose.covariance[7] = 0.1   # y
        odom.pose.covariance[14] = 0.1  # z
        odom.pose.covariance[21] = 0.1  # roll
        odom.pose.covariance[28] = 0.1  # pitch
        odom.pose.covariance[35] = 0.1  # yaw

        self.odom_pub.publish(odom)

    def _publish_deskewed_points(self, deskewed_frame: np.ndarray, stamp):
        """Publish the deskewed point cloud."""
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = self.base_frame
        deskewed_cloud = pc2.create_cloud_xyz32(header, deskewed_frame[:, :3])
        self.deskewed_pub.publish(deskewed_cloud)

    def destroy_node(self):
        self.get_logger().info("Shutting down odometry node.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

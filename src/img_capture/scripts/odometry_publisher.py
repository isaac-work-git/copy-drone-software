#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from px4_msgs.msg import VehicleAttitude, SensorGps
from std_msgs.msg import String
import json
import numpy as np

class PoseFilter:
  def __init__(self, max_ref_count=10, alpha_alt=0.5, alpha_enu=0.9, alpha_rpy=0.5):
    self.max_ref_count = max_ref_count
    self.alpha_alt = alpha_alt
    self.alpha_enu = alpha_enu
    self.alpha_rpy = alpha_rpy

    # Running sums for reference
    self.ref_count = 0
    self.sum_lat = 0.0
    self.sum_lon = 0.0
    self.sum_alt = 0.0

    # Final reference (lat, lon, alt)
    self.ref_lat = None
    self.ref_lon = None
    self.ref_alt = None

    # ECEF of reference
    self.ref_ecef = None

    # Precomputed ENU transform unit vectors
    self.e_hat = None
    self.n_hat = None
    self.u_hat = None

    # Store current values
    self.timestamp = None
    self.alt = None
    self.x = None
    self.y = None
    self.z = None
    self.pitch = None
    self.roll = None
    self.yaw = None

  @staticmethod
  def _wgs84_to_ecef(lat_deg, lon_deg, alt_m):
    # WGS84 constants
    a = 6378137.0             # semi-major axis
    f = 1.0 / 298.257223563   # flattening
    b = a * (1 - f)           # semi-minor axis
    e_sq = (a**2 - b**2) / a**2

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
    X = (N + alt_m) * np.cos(lat) * np.cos(lon)
    Y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e_sq) + alt_m) * np.sin(lat)
    return np.array([X, Y, Z])

  def _compute_ref_ecef_and_enu(self):
    self.ref_ecef = self._wgs84_to_ecef(self.ref_lat, self.ref_lon, self.ref_alt)

    # For ENU, define unit vectors at the reference lat/lon
    lat0 = np.radians(self.ref_lat)
    lon0 = np.radians(self.ref_lon)

    self.e_hat = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    self.n_hat = np.array([
        -np.sin(lat0)*np.cos(lon0),
        -np.sin(lat0)*np.sin(lon0),
          np.cos(lat0)
    ])
    self.u_hat = np.array([
        np.cos(lat0)*np.cos(lon0),
        np.cos(lat0)*np.sin(lon0),
        np.sin(lat0)
    ])

  def _update_reference(self, lat, lon, alt):
    if self.ref_count < self.max_ref_count:
        self.ref_count += 1
        self.sum_lat += lat
        self.sum_lon += lon
        self.sum_alt += alt
        # partial average
        self.ref_lat = self.sum_lat / self.ref_count
        self.ref_lon = self.sum_lon / self.ref_count
        self.ref_alt = self.sum_alt / self.ref_count
        # Recompute ECEF of reference and ENU transform
        self._compute_ref_ecef_and_enu()
    elif self.ref_lat is None:
        # If code started with max_ref_count=0 or something, we set once
        self.ref_lat = lat
        self.ref_lon = lon
        self.ref_alt = alt
        self.ref_count = self.max_ref_count
        self._compute_ref_ecef_and_enu()
    else:
        # Do nothing => reference is frozen
        pass

  def _latlonalt_to_enu(self, lat, lon, alt):
    if self.ref_ecef is None:
        # no reference yet => return zeros
        return (0.0, 0.0, 0.0)
    xyz = self._wgs84_to_ecef(lat, lon, alt)
    dx = xyz[0] - self.ref_ecef[0]
    dy = xyz[1] - self.ref_ecef[1]
    dz = xyz[2] - self.ref_ecef[2]
    dvec = np.array([dx, dy, dz])
    x_e = np.dot(self.e_hat, dvec)
    y_e = np.dot(self.n_hat, dvec)
    z_e = np.dot(self.u_hat, dvec)
    return (x_e, y_e, z_e)

  @staticmethod
  def _quat_to_euler_ned(q):
    q0, q1, q2, q3 = q
    # Roll
    sinr_cosp = 2.0 * (q0*q1 + q2*q3)
    cosr_cosp = 1.0 - 2.0 * (q1**2 + q2**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2.0 * (q0*q2 - q3*q1)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw
    siny_cosp = 2.0 * (q0*q3 + q1*q2)
    cosy_cosp = 1.0 - 2.0 * (q2**2 + q3**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)

  @staticmethod
  def _angle_wrap(a):
    while a >= np.pi:
      a -= 2.0*np.pi
    while a < -np.pi:
      a += 2.0*np.pi
    return a

  def _euler_ned_to_enu(self, roll_ned, pitch_ned, yaw_ned):
    # 1) Yaw shift: yaw_enu = yaw_ned - 90 degrees
    yaw_enu = yaw_ned - np.pi / 2
    # 2) Wrap yaw to [-pi, pi)
    yaw_enu = self._angle_wrap(yaw_enu)
    # 3) Roll and pitch remain the same
    roll_enu = roll_ned
    pitch_enu = pitch_ned
    return roll_enu, pitch_enu, yaw_enu

  @staticmethod
  def _rotmat_to_euler_enu(R):
    # roll = atan2(R[2,1], R[2,2])
    # pitch= asin(-R[2,0])
    # yaw  = atan2(R[1,0], R[0,0])
    roll = np.arctan2(R[2,1], R[2,2])
    sinp = -R[2,0]
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(R[1,0], R[0,0])
    return (roll, pitch, yaw)

  def _alpha_angle(self, alpha, angle_in, angle_prev):
    # Shift angle_in near angle_prev
    diff = angle_in - angle_prev
    diff_wrapped = self._angle_wrap(diff)
    angle_in_shifted = angle_prev + diff_wrapped
    angle_out = alpha*angle_in_shifted + (1.0 - alpha)*angle_prev
    return self._angle_wrap(angle_out)

  def process_coords(self, timestamp, lat, lon, alt):
    # Update reference if needed
    self._update_reference(lat, lon, alt)

    # Filter altitude
    if self.alt is None:
      new_alt = alt
    else:
      new_alt = self.alpha_alt * alt + (1.0 - self.alpha_alt) * self.alt

    # Convert lat/lon/(filtered alt) -> ENU (x,y,z)
    x_e, y_e, z_e = self._latlonalt_to_enu(lat, lon, new_alt)

    # Alpha-filter x, y, and z
    if self.x is None:
      new_x = x_e
      new_y = y_e
      new_z = z_e
    else:
      new_x = self.alpha_enu * x_e + (1.0 - self.alpha_enu)*self.x
      new_y = self.alpha_enu * y_e  + (1.0 - self.alpha_enu)*self.y
      new_z = self.alpha_enu * z_e  + (1.0 - self.alpha_enu)*self.z

    # Update everything
    self.timestamp = timestamp
    self.alt = new_alt
    self.x = new_x
    self.y = new_y
    self.z = new_z

  def process_quat(self, q_w, q_x, q_y, q_z):

    # Convert quaternion body->NED to Euler NED => then to Euler ENU
    roll_ned, pitch_ned, yaw_ned = self._quat_to_euler_ned([q_w, q_x, q_y, q_z])
    roll_enu_raw, pitch_enu_raw, yaw_enu_raw = self._euler_ned_to_enu(roll_ned, pitch_ned, yaw_ned)

    # Alpha-filter pitch, roll, and yaw
    if self.pitch is None:
      new_pitch = pitch_enu_raw
      new_roll  = roll_enu_raw
      new_yaw   = yaw_enu_raw
    else:
      new_pitch = self._alpha_angle(self.alpha_rpy, pitch_enu_raw, self.pitch)
      new_roll  = self._alpha_angle(self.alpha_rpy, roll_enu_raw, self.roll)
      new_yaw  = self._alpha_angle(self.alpha_rpy, yaw_enu_raw, self.yaw)

    # Update
    self.pitch = new_pitch
    self.roll = new_roll
    self.yaw = new_yaw

  def get_current(self):
    return {"timestamp":self.timestamp,
            "x":self.x,
            "y":self.y,
            "z":self.z,
            "pitch":self.pitch,
            "roll":self.roll,
            "yaw":self.yaw,
            "ref_lat":self.ref_lat,
            "ref_lon":self.ref_lon,
            "ref_alt":self.ref_alt}

class CombinedOdometryPublisher(Node):
    def __init__(self):
        super().__init__('combined_odometry_publisher')
        self.get_logger().info("Combined Odometry Publisher started.\n")

        # Save filter
        self.filter = PoseFilter(alpha_alt=0.1, alpha_enu=0.9, alpha_rpy=0.1)
        self.quat_updated = False
        self.coords_updated = False

        # Set up QoS profile for PX4 topics (BEST_EFFORT)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Subscriptions
        self.create_subscription(VehicleAttitude,
                                 '/fmu/out/vehicle_attitude',
                                 self.attitude_callback,
                                 qos_profile)

        self.create_subscription(SensorGps,
                                 '/fmu/out/vehicle_gps_position',
                                 self.gps_callback,
                                 qos_profile)

        # Publisher: combined odometry data
        self.publisher = self.create_publisher(String, 'combined_odometry', 10)

        # Timer for periodic publishing (20 Hz)
        self.timer = self.create_timer(1.0 / 20.0, self.timer_callback)

    def attitude_callback(self, msg: VehicleAttitude):
        """
        Callback for VehicleAttitude messages to retrieve raw quaternion (body->NED).
        """
        # self.get_logger().info("Attitude Callback\n")
        # Ensure we cast each to Python float
        q = msg.q  # [w, x, y, z]
        qw = float(q[0])
        qx = float(q[1])
        qy = float(q[2])
        qz = float(q[3])
        self.filter.process_quat(qw, qx, qy, qz)
        self.quat_updated = True

    def gps_callback(self, msg: SensorGps):
        """
        Callback for SensorGps messages to retrieve raw data:
        timestamp, lat, lon, alt ellipsoid, velocities, accuracy, etc.
        """
        # self.get_logger().info("GPS Callback\n")

        timestamp = int(msg.timestamp)  # microseconds (as int)
        latitude = float(msg.latitude_deg)
        longitude = float(msg.longitude_deg)
        altitude_ellipsoid = float(msg.altitude_ellipsoid_m)

        self.filter.process_coords(timestamp, latitude, longitude, altitude_ellipsoid)
        self.coords_updated = True

        

    def timer_callback(self):
        """
        Periodic publish of combined data as a JSON string.
        """
        if self.coords_updated and self.quat_updated:
            self.coords_updated = False
            self.quat_updated = False


            data_dict = self.filter.get_current()

            # Convert dict to JSON
            json_msg = String()
            json_msg.data = json.dumps(data_dict)

            # Publish
            self.publisher.publish(json_msg)

            # Log
            self.get_logger().info(
                f"GPS timestamp={data_dict['timestamp']}, "
                f"x={data_dict['x']}, y={data_dict['y']}, z={data_dict['z']}, "
                f"pitch={data_dict['pitch']}, roll={data_dict['roll']}, yaw={data_dict['yaw']}, "
                f"ref_lat={data_dict['ref_lat']}, ref_lon={data_dict['ref_lon']}, ref_alt={data_dict['ref_alt']}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = CombinedOdometryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


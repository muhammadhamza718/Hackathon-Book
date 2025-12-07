---
title: 'Integrating Real-World Sensors into Simulation'
description: 'Integrating real-world sensor data into simulation environments, explaining bridging real-world sensor data and simulation, using recorded data or live feeds'
chapter: 7
lesson: 1
module: 2
sidebar_label: 'Integrating Real-World Sensors into Simulation'
sidebar_position: 1
tags: ['Sensors', 'Simulation', 'Real-World Integration', 'Data Bridging', 'Sensor Fusion']
keywords: ['sensor integration', 'real-world data', 'simulation', 'data bridging', 'live feeds', 'recorded data', 'sensor fusion']
---

# Integrating Real-World Sensors into Simulation

## Overview

Integrating real-world sensor data into simulation environments is a crucial capability for robotics development, enabling mixed reality scenarios, validation of algorithms against real data, and hybrid testing approaches. This lesson covers techniques for bridging real-world sensor data with simulation, including methods for using recorded data or live sensor feeds in simulation environments.

## Real-World Sensor Integration Concepts

### Mixed Reality in Robotics

Mixed reality in robotics combines real-world sensor data with virtual environments to create hybrid simulation experiences. This approach offers several benefits:

- **Validation**: Test algorithms against real-world conditions
- **Safety**: Evaluate systems without real-world risk
- **Cost Reduction**: Reduce physical testing requirements
- **Flexibility**: Modify environment parameters while using real sensor data
- **Training**: Provide realistic training scenarios

### Types of Sensor Integration

There are several approaches to integrating real-world sensors:

1. **Offline Integration**: Using recorded sensor data
2. **Online Integration**: Using live sensor feeds
3. **Hybrid Integration**: Combining real and synthetic sensors
4. **Augmented Simulation**: Enhancing virtual sensors with real data

## Data Bridging Techniques

### ROS Bag Integration

ROS bags are the standard format for storing sensor data in ROS ecosystems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import Header
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import subprocess

class SensorDataBridge(Node):
    def __init__(self):
        super().__init__('sensor_data_bridge')

        # Publishers for real-world sensor data
        self.image_pub = self.create_publisher(Image, '/real_world/camera/image_raw', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/real_world/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/real_world/imu/data', 10)

        # Parameters
        self.declare_parameter('bag_path', '')
        self.declare_parameter('start_time', 0.0)
        self.declare_parameter('playback_speed', 1.0)

        # Playback control
        self.bag_path = self.get_parameter('bag_path').value
        self.start_time = self.get_parameter('start_time').value
        self.playback_speed = self.get_parameter('playback_speed').value

        # Timer for processing
        self.timer = self.create_timer(0.01, self.process_bag_data)

        # Bag reader
        self.bag_reader = None
        self.initialize_bag_reader()

    def initialize_bag_reader(self):
        """Initialize the bag reader for real-world sensor data"""
        try:
            # Create storage options
            storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')

            # Create converter options
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )

            # Create bag reader
            self.bag_reader = rosbag2_py.SequentialReader()
            self.bag_reader.open(storage_options, converter_options)

            # Get topics and types
            topics_types = self.bag_reader.get_all_topics_and_types()
            self.topic_types_map = {topic.topic_metadata.name: topic.topic_metadata.type for topic in topics_types}

            self.get_logger().info(f'Initialized bag reader for: {self.bag_path}')
            self.get_logger().info(f'Topics: {list(self.topic_types_map.keys())}')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize bag reader: {e}')

    def process_bag_data(self):
        """Process data from the bag file"""
        if self.bag_reader is None:
            return

        try:
            # Read next message from bag
            if self.bag_reader.has_next():
                topic_name, msg_data, timestamp = self.bag_reader.read_next()

                # Convert message based on type
                msg_type = self.topic_types_map.get(topic_name)
                if msg_type:
                    # Deserialize the message
                    msg_class = get_message(msg_type)
                    msg = deserialize_message(msg_data, msg_class)

                    # Republish to simulation topics
                    self.republish_message(topic_name, msg)

        except Exception as e:
            self.get_logger().error(f'Error processing bag data: {e}')

    def republish_message(self, topic_name, msg):
        """Republish message to simulation topics"""
        # Handle different sensor types
        if 'image' in topic_name.lower():
            self.image_pub.publish(msg)
        elif 'scan' in topic_name.lower() or 'lidar' in topic_name.lower():
            self.scan_pub.publish(msg)
        elif 'imu' in topic_name.lower():
            self.imu_pub.publish(msg)
        else:
            # Generic republishing for other topics
            self.get_logger().info(f'Received message from topic: {topic_name}')

    def start_playback(self):
        """Start playing back the bag file"""
        if self.bag_reader:
            self.get_logger().info('Starting bag playback...')
            # Implementation would include time synchronization
        else:
            self.get_logger().error('Bag reader not initialized')

    def stop_playback(self):
        """Stop bag playback"""
        self.get_logger().info('Stopping bag playback...')
        # Implementation would include cleanup

def main(args=None):
    rclpy.init(args=args)
    sensor_bridge = SensorDataBridge()

    try:
        rclpy.spin(sensor_bridge)
    except KeyboardInterrupt:
        sensor_bridge.get_logger().info('Shutting down sensor data bridge...')
    finally:
        sensor_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Live Sensor Data Integration

For real-time integration with live sensors:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
import threading
import time

class LiveSensorIntegrator(Node):
    def __init__(self):
        super().__init__('live_sensor_integrator')

        # Real sensor subscribers
        self.real_image_sub = self.create_subscription(
            Image, '/real/camera/image_raw', self.real_image_callback, 10
        )
        self.real_scan_sub = self.create_subscription(
            LaserScan, '/real/scan', self.real_scan_callback, 10
        )
        self.real_imu_sub = self.create_subscription(
            Imu, '/real/imu/data', self.real_imu_callback, 10
        )

        # Simulation publishers
        self.sim_image_pub = self.create_publisher(Image, '/sim/camera/image_raw', 10)
        self.sim_scan_pub = self.create_publisher(LaserScan, '/sim/scan', 10)
        self.sim_imu_pub = self.create_publisher(Imu, '/sim/imu/data', 10)

        # Data buffers for synchronization
        self.image_buffer = []
        self.scan_buffer = []
        self.imu_buffer = []

        # Buffer sizes
        self.buffer_size = 10

        # Lock for thread safety
        self.buffer_lock = threading.Lock()

    def real_image_callback(self, msg):
        """Callback for real camera data"""
        with self.buffer_lock:
            self.image_buffer.append(msg)
            if len(self.image_buffer) > self.buffer_size:
                self.image_buffer.pop(0)

        # Publish to simulation
        self.sim_image_pub.publish(msg)

    def real_scan_callback(self, msg):
        """Callback for real LiDAR data"""
        with self.buffer_lock:
            self.scan_buffer.append(msg)
            if len(self.scan_buffer) > self.buffer_size:
                self.scan_buffer.pop(0)

        # Publish to simulation
        self.sim_scan_pub.publish(msg)

    def real_imu_callback(self, msg):
        """Callback for real IMU data"""
        with self.buffer_lock:
            self.imu_buffer.append(msg)
            if len(self.imu_buffer) > self.buffer_size:
                self.imu_buffer.pop(0)

        # Publish to simulation
        self.sim_imu_pub.publish(msg)

    def get_latest_image(self):
        """Get the latest image from buffer"""
        with self.buffer_lock:
            if self.image_buffer:
                return self.image_buffer[-1]
            return None

    def get_synchronized_data(self, timestamp_tolerance=0.01):
        """Get synchronized sensor data"""
        with self.buffer_lock:
            # Find data closest to the current time within tolerance
            current_time = self.get_clock().now()

            # Look for synchronized data
            for i in range(len(self.image_buffer)):
                img_time = rclpy.time.Time.from_msg(self.image_buffer[i].header.stamp)
                for j in range(len(self.scan_buffer)):
                    scan_time = rclpy.time.Time.from_msg(self.scan_buffer[j].header.stamp)
                    if abs((img_time - scan_time).nanoseconds) < timestamp_tolerance * 1e9:
                        # Found synchronized image and scan
                        for k in range(len(self.imu_buffer)):
                            imu_time = rclpy.time.Time.from_msg(self.imu_buffer[k].header.stamp)
                            if abs((img_time - imu_time).nanoseconds) < timestamp_tolerance * 1e9:
                                return {
                                    'image': self.image_buffer[i],
                                    'scan': self.scan_buffer[j],
                                    'imu': self.imu_buffer[k]
                                }
        return None
```

## Sensor Data Preprocessing

### Data Filtering and Calibration

Real-world sensor data often requires preprocessing before integration:

```python
import numpy as np
from scipy import ndimage
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2

class SensorDataPreprocessor:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.scan_filter_params = {
            'min_range': 0.1,
            'max_range': 30.0,
            'median_filter_window': 3
        }
        self.image_filter_params = {
            'gaussian_blur_kernel': (5, 5),
            'gaussian_sigma': 1.0
        }

    def preprocess_lidar_data(self, scan_msg):
        """Preprocess LiDAR data for simulation"""
        # Convert to numpy array
        ranges = np.array(scan_msg.ranges)

        # Filter out invalid ranges
        ranges = np.where(
            (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max),
            ranges,
            np.inf
        )

        # Apply median filter to reduce noise
        if len(ranges) > self.scan_filter_params['median_filter_window']:
            filtered_ranges = ndimage.median_filter(
                ranges,
                size=self.scan_filter_params['median_filter_window']
            )
        else:
            filtered_ranges = ranges

        # Create new message with filtered data
        filtered_scan = LaserScan()
        filtered_scan.header = scan_msg.header
        filtered_scan.angle_min = scan_msg.angle_min
        filtered_scan.angle_max = scan_msg.angle_max
        filtered_scan.angle_increment = scan_msg.angle_increment
        filtered_scan.time_increment = scan_msg.time_increment
        filtered_scan.scan_time = scan_msg.scan_time
        filtered_scan.range_min = scan_msg.range_min
        filtered_scan.range_max = scan_msg.range_max
        filtered_scan.ranges = filtered_ranges.tolist()
        filtered_scan.intensities = scan_msg.intensities  # Copy intensities if available

        return filtered_scan

    def preprocess_camera_data(self, image_msg):
        """Preprocess camera data for simulation"""
        # Convert ROS image to OpenCV
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Apply Gaussian blur for noise reduction
        blurred_image = cv2.GaussianBlur(
            cv_image,
            self.image_filter_params['gaussian_blur_kernel'],
            self.image_filter_params['gaussian_sigma']
        )

        # Convert back to ROS image
        processed_image_msg = self.cv_bridge.cv2_to_imgmsg(blurred_image, encoding='bgr8')
        processed_image_msg.header = image_msg.header

        return processed_image_msg

    def calibrate_sensor_data(self, sensor_msg, calibration_matrix):
        """Apply calibration to sensor data"""
        # Apply calibration based on sensor type
        if isinstance(sensor_msg, LaserScan):
            return self.calibrate_lidar_data(sensor_msg, calibration_matrix)
        elif isinstance(sensor_msg, Image):
            return self.calibrate_camera_data(sensor_msg, calibration_matrix)
        else:
            return sensor_msg

    def calibrate_lidar_data(self, scan_msg, calibration_matrix):
        """Apply LiDAR calibration"""
        # In this example, calibration might involve adjusting ranges
        # based on known systematic errors
        calibrated_ranges = []
        for i, range_val in enumerate(scan_msg.ranges):
            if np.isfinite(range_val):
                # Apply calibration factor based on angle
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                calibration_factor = self.calculate_calibration_factor(angle, calibration_matrix)
                calibrated_ranges.append(range_val * calibration_factor)
            else:
                calibrated_ranges.append(range_val)

        scan_msg.ranges = calibrated_ranges
        return scan_msg

    def calculate_calibration_factor(self, angle, calibration_matrix):
        """Calculate calibration factor based on angle and calibration matrix"""
        # This is a simplified example
        # Real calibration would use more sophisticated methods
        return 1.0 + 0.01 * np.sin(angle)  # Example calibration function
```

## Time Synchronization

### Timestamp Alignment

Proper time synchronization is crucial for sensor integration:

```python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import Header
import time

class TimeSynchronizer(Node):
    def __init__(self):
        super().__init__('time_synchronizer')

        # Create subscribers for different sensor types
        self.image_sub = self.create_subscription(
            Image, '/real/camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/real/scan', self.scan_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/real/imu/data', self.imu_callback, 10
        )

        # Publishers for synchronized data
        self.sync_image_pub = self.create_publisher(Image, '/sync/camera/image_raw', 10)
        self.sync_scan_pub = self.create_publisher(LaserScan, '/sync/scan', 10)
        self.sync_imu_pub = self.create_publisher(Imu, '/sync/imu/data', 10)

        # Time alignment parameters
        self.time_offset = 0.0  # Offset to align real and sim time
        self.timestamp_tolerance = 0.05  # 50ms tolerance

        # Storage for buffered data
        self.image_buffer = []
        self.scan_buffer = []
        self.imu_buffer = []

    def image_callback(self, msg):
        """Handle incoming image messages"""
        # Adjust timestamp to align with simulation
        aligned_time = self.align_timestamp(msg.header.stamp)
        msg.header.stamp = aligned_time

        # Store in buffer
        self.image_buffer.append(msg)
        self.prune_old_messages(self.image_buffer)

        # Try to synchronize with other sensors
        self.attempt_synchronization()

    def scan_callback(self, msg):
        """Handle incoming scan messages"""
        # Adjust timestamp to align with simulation
        aligned_time = self.align_timestamp(msg.header.stamp)
        msg.header.stamp = aligned_time

        # Store in buffer
        self.scan_buffer.append(msg)
        self.prune_old_messages(self.scan_buffer)

        # Try to synchronize with other sensors
        self.attempt_synchronization()

    def imu_callback(self, msg):
        """Handle incoming IMU messages"""
        # Adjust timestamp to align with simulation
        aligned_time = self.align_timestamp(msg.header.stamp)
        msg.header.stamp = aligned_time

        # Store in buffer
        self.imu_buffer.append(msg)
        self.prune_old_messages(self.imu_buffer)

        # Try to synchronize with other sensors
        self.attempt_synchronization()

    def align_timestamp(self, timestamp):
        """Align real-world timestamp with simulation time"""
        # Convert ROS time to seconds
        real_time_seconds = timestamp.sec + timestamp.nanosec / 1e9

        # Apply time offset
        aligned_time_seconds = real_time_seconds + self.time_offset

        # Convert back to ROS time format
        aligned_time = Time()
        aligned_time.sec = int(aligned_time_seconds)
        aligned_time.nanosec = int((aligned_time_seconds - int(aligned_time_seconds)) * 1e9)

        return aligned_time

    def attempt_synchronization(self):
        """Attempt to synchronize sensor data based on timestamps"""
        # Find closest timestamps across all sensor types
        sync_data = self.find_synchronized_data()

        if sync_data:
            # Publish synchronized data
            if sync_data.get('image'):
                self.sync_image_pub.publish(sync_data['image'])
            if sync_data.get('scan'):
                self.sync_scan_pub.publish(sync_data['scan'])
            if sync_data.get('imu'):
                self.sync_imu_pub.publish(sync_data['imu'])

    def find_synchronized_data(self):
        """Find synchronized data across sensor types"""
        if not (self.image_buffer and self.scan_buffer and self.imu_buffer):
            return None

        # Find the most recent timestamp
        latest_image_time = self.image_buffer[-1].header.stamp
        latest_scan_time = self.scan_buffer[-1].header.stamp
        latest_imu_time = self.imu_buffer[-1].header.stamp

        # Find data within tolerance
        sync_data = {}

        # Find closest image to scan time
        closest_image = self.find_closest_message(
            self.image_buffer, latest_scan_time, self.timestamp_tolerance
        )
        if closest_image:
            sync_data['image'] = closest_image

        # Find closest scan to image time
        closest_scan = self.find_closest_message(
            self.scan_buffer, latest_image_time, self.timestamp_tolerance
        )
        if closest_scan:
            sync_data['scan'] = closest_scan

        # Find closest IMU to image time
        closest_imu = self.find_closest_message(
            self.imu_buffer, latest_image_time, self.timestamp_tolerance
        )
        if closest_imu:
            sync_data['imu'] = closest_imu

        return sync_data if len(sync_data) == 3 else None

    def find_closest_message(self, buffer, target_time, tolerance):
        """Find the closest message to target time within tolerance"""
        if not buffer:
            return None

        target_ts = target_time.sec + target_time.nanosec / 1e9

        closest_msg = None
        min_diff = float('inf')

        for msg in buffer:
            msg_ts = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            diff = abs(msg_ts - target_ts)

            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_msg = msg

        return closest_msg

    def prune_old_messages(self, buffer):
        """Remove old messages from buffer to manage memory"""
        current_time = self.get_clock().now().seconds_nanoseconds()
        current_ts = current_time[0] + current_time[1] / 1e9

        # Keep only messages from the last 2 seconds
        cutoff_time = current_ts - 2.0

        # Remove old messages
        buffer[:] = [msg for msg in buffer if
                     msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9 > cutoff_time]
```

## Coordinate Frame Integration

### TF (Transform) Integration

Proper coordinate frame handling is essential for sensor integration:

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import Image, LaserScan
import tf2_geometry_msgs
import tf2_ros

class SensorFrameIntegrator(Node):
    def __init__(self):
        super().__init__('sensor_frame_integrator')

        # Initialize TF broadcaster and listener
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers for sensor data
        self.image_sub = self.create_subscription(
            Image, '/real/camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/real/scan', self.scan_callback, 10
        )

        # Publishers for transformed data
        self.transformed_image_pub = self.create_publisher(Image, '/transformed/camera/image_raw', 10)
        self.transformed_scan_pub = self.create_publisher(LaserScan, '/transformed/scan', 10)

        # Timer for broadcasting transforms
        self.timer = self.create_timer(0.01, self.broadcast_transforms)

    def broadcast_transforms(self):
        """Broadcast coordinate transforms"""
        # Example: Robot base to camera transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'

        # Set transform (example values)
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    def image_callback(self, msg):
        """Handle image with coordinate frame transformation"""
        try:
            # Transform image coordinates if needed
            # This is a simplified example
            transformed_msg = msg
            transformed_msg.header.frame_id = 'sim_camera_frame'  # Change to simulation frame

            self.transformed_image_pub.publish(transformed_msg)
        except Exception as e:
            self.get_logger().warn(f'Error transforming image: {e}')

    def scan_callback(self, msg):
        """Handle LiDAR scan with coordinate frame transformation"""
        try:
            # Look up transform from sensor frame to simulation frame
            transform = self.tf_buffer.lookup_transform(
                'sim_base_link',  # Target frame in simulation
                msg.header.frame_id,  # Source frame (real sensor)
                rclpy.time.Time(),  # Latest available
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Apply transform to scan data
            # Note: This is a simplified example - real implementation would
            # transform individual scan points based on the transform
            transformed_msg = msg
            transformed_msg.header.frame_id = 'sim_base_link'

            # In real implementation, you would transform each range reading
            # based on the sensor's position and orientation
            self.transformed_scan_pub.publish(transformed_msg)

        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Could not transform scan: {ex}')
```

## Sensor Fusion Integration

### Combining Real and Virtual Sensors

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import Float32
import numpy as np

class SensorFusionIntegrator(Node):
    def __init__(self):
        super().__init__('sensor_fusion_integrator')

        # Real sensor subscribers
        self.real_image_sub = self.create_subscription(
            Image, '/real/camera/image_raw', self.real_image_callback, 10
        )
        self.real_scan_sub = self.create_subscription(
            LaserScan, '/real/scan', self.real_scan_callback, 10
        )

        # Virtual sensor subscribers (from simulation)
        self.virtual_image_sub = self.create_subscription(
            Image, '/sim/camera/image_raw', self.virtual_image_callback, 10
        )
        self.virtual_scan_sub = self.create_subscription(
            LaserScan, '/sim/scan', self.virtual_scan_callback, 10
        )

        # Fusion publishers
        self.fused_image_pub = self.create_publisher(Image, '/fused/camera/image_raw', 10)
        self.fused_scan_pub = self.create_publisher(LaserScan, '/fused/scan', 10)
        self.confidence_pub = self.create_publisher(Float32, '/fusion/confidence', 10)

        # Fusion parameters
        self.confidence_threshold = 0.7
        self.fusion_strategy = 'confidence_based'  # or 'weighted_average'

    def real_image_callback(self, msg):
        """Process real image data"""
        # Store real image with timestamp
        self.process_real_sensor_data('image', msg)

    def virtual_image_callback(self, msg):
        """Process virtual image data"""
        # Store virtual image with timestamp
        self.process_virtual_sensor_data('image', msg)

    def real_scan_callback(self, msg):
        """Process real LiDAR data"""
        # Store real scan with timestamp
        self.process_real_sensor_data('scan', msg)

    def virtual_scan_callback(self, msg):
        """Process virtual LiDAR data"""
        # Store virtual scan with timestamp
        self.process_virtual_sensor_data('scan', msg)

    def process_real_sensor_data(self, sensor_type, msg):
        """Process real sensor data"""
        # Store in real data buffer
        # Implement fusion logic based on confidence
        if sensor_type == 'image':
            self.perform_image_fusion(msg)
        elif sensor_type == 'scan':
            self.perform_scan_fusion(msg)

    def process_virtual_sensor_data(self, sensor_type, msg):
        """Process virtual sensor data"""
        # Store in virtual data buffer
        # Implement fusion logic based on confidence
        pass

    def perform_image_fusion(self, real_image):
        """Perform image fusion between real and virtual"""
        # Example: Overlay virtual objects on real image
        # This is a simplified example
        fused_image = real_image  # In practice, blend with virtual elements

        # Calculate confidence based on sensor quality
        confidence = self.calculate_image_confidence(real_image)

        # Publish confidence
        confidence_msg = Float32()
        confidence_msg.data = confidence
        self.confidence_pub.publish(confidence_msg)

        # Publish fused image
        self.fused_image_pub.publish(fused_image)

    def perform_scan_fusion(self, real_scan):
        """Perform LiDAR scan fusion between real and virtual"""
        # Example: Merge real and virtual scan data
        # This is a simplified example

        # Calculate confidence in real data
        confidence = self.calculate_scan_confidence(real_scan)

        # In practice, you would:
        # 1. Transform virtual scan to real sensor frame
        # 2. Merge overlapping regions based on confidence
        # 3. Keep high-confidence real data, supplement with virtual

        fused_scan = real_scan  # Placeholder

        # Publish confidence
        confidence_msg = Float32()
        confidence_msg.data = confidence
        self.confidence_pub.publish(confidence_msg)

        # Publish fused scan
        self.fused_scan_pub.publish(fused_scan)

    def calculate_image_confidence(self, image_msg):
        """Calculate confidence in image data"""
        # Example: Confidence based on image quality metrics
        # This is a simplified example
        return 0.9  # High confidence for real data

    def calculate_scan_confidence(self, scan_msg):
        """Calculate confidence in scan data"""
        # Example: Confidence based on range and quality
        valid_ranges = [r for r in scan_msg.ranges if r != float('inf') and r != float('-inf')]
        if not valid_ranges:
            return 0.0

        # Confidence based on number of valid readings
        confidence = len(valid_ranges) / len(scan_msg.ranges)

        # Adjust based on range values
        avg_range = sum(valid_ranges) / len(valid_ranges)
        if avg_range > 10.0:  # Far distances have lower confidence
            confidence *= 0.8

        return min(confidence, 1.0)
```

## Performance Considerations

### Data Throughput Optimization

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from threading import Lock
import time

class OptimizedSensorBridge(Node):
    def __init__(self):
        super().__init__('optimized_sensor_bridge')

        # Configure QoS for different sensor types
        # High-frequency sensors (LiDAR) - use best effort
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Lower-frequency sensors (cameras) - use reliable
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers with optimized QoS
        self.real_image_sub = self.create_subscription(
            Image, '/real/camera/image_raw', self.optimized_image_callback, camera_qos
        )
        self.real_scan_sub = self.create_subscription(
            LaserScan, '/real/scan', self.optimized_scan_callback, lidar_qos
        )

        # Publishers with optimized QoS
        self.sim_image_pub = self.create_publisher(Image, '/sim/camera/image_raw', camera_qos)
        self.sim_scan_pub = self.create_publisher(LaserScan, '/sim/scan', lidar_qos)

        # Performance monitoring
        self.message_count = 0
        self.start_time = time.time()

        # Processing optimization
        self.processing_interval = 0.01  # 100 Hz
        self.last_process_time = 0.0

        # Thread safety
        self.data_lock = Lock()

    def optimized_image_callback(self, msg):
        """Optimized image callback with processing throttling"""
        current_time = time.time()

        # Throttle processing to avoid overwhelming system
        if current_time - self.last_process_time < self.processing_interval:
            return

        with self.data_lock:
            # Process and forward image
            self.process_and_forward_image(msg)
            self.last_process_time = current_time
            self.message_count += 1

    def optimized_scan_callback(self, msg):
        """Optimized scan callback with processing throttling"""
        current_time = time.time()

        # Throttle processing
        if current_time - self.last_process_time < self.processing_interval:
            return

        with self.data_lock:
            # Process and forward scan
            self.process_and_forward_scan(msg)
            self.last_process_time = current_time
            self.message_count += 1

    def process_and_forward_image(self, msg):
        """Process and forward image with optimization"""
        # Forward to simulation
        self.sim_image_pub.publish(msg)

    def process_and_forward_scan(self, msg):
        """Process and forward scan with optimization"""
        # Forward to simulation
        self.sim_scan_pub.publish(msg)

    def report_performance(self):
        """Report performance metrics"""
        elapsed_time = time.time() - self.start_time
        rate = self.message_count / elapsed_time if elapsed_time > 0 else 0

        self.get_logger().info(
            f'Processed {self.message_count} messages in {elapsed_time:.2f}s '
            f'at rate {rate:.2f} msg/s'
        )
```

## Best Practices for Real-World Sensor Integration

### 1. Data Validation
- Validate sensor data ranges and formats
- Check for missing or corrupted data
- Implement data quality metrics
- Log anomalies for debugging

### 2. Synchronization
- Implement proper time synchronization
- Use appropriate buffer sizes
- Handle different sensor frequencies
- Maintain temporal consistency

### 3. Performance
- Optimize data processing pipelines
- Use appropriate QoS settings
- Implement data compression if needed
- Monitor system performance

### 4. Error Handling
- Handle sensor failures gracefully
- Implement fallback behaviors
- Provide diagnostic information
- Log errors for troubleshooting

## Learning Objectives

By the end of this lesson, you should be able to:
- Integrate real-world sensor data into simulation environments
- Bridge real-world sensor data with simulation systems
- Use recorded data (ROS bags) or live sensor feeds
- Implement time synchronization for multi-sensor systems
- Apply preprocessing and calibration to sensor data
- Handle coordinate frame transformations
- Implement sensor fusion between real and virtual data
- Apply best practices for performance and reliability
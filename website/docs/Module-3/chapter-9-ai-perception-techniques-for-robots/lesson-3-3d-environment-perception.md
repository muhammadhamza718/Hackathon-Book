---
title: '3D Environment Perception'
description: 'Explore 3D environment perception for robots, covering point cloud processing, depth image analysis, SLAM basics'
chapter: 9
lesson: 3
module: 3
sidebar_label: '3D Environment Perception'
sidebar_position: 3
tags: ['3D Perception', 'Point Clouds', 'SLAM', 'Depth Analysis', 'Environment Mapping']
keywords: ['3D perception', 'point clouds', 'depth analysis', 'SLAM', 'environment mapping', 'robot navigation', 'spatial understanding']
---

# 3D Environment Perception

## Overview

3D environment perception is fundamental for robotic autonomy, enabling robots to understand and navigate through three-dimensional spaces. This lesson covers essential techniques for processing 3D sensor data including point clouds, depth images, and Simultaneous Localization and Mapping (SLAM) systems. These capabilities allow robots to build spatial representations of their environment and localize themselves within it.

## Point Cloud Processing

### Point Cloud Fundamentals

A point cloud is a collection of data points in 3D space, typically obtained from LiDAR sensors, stereo cameras, or structured light systems. Each point has coordinates (x, y, z) and may include additional attributes like color, intensity, or normal vectors.

```python
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class PointCloudProcessor:
    def __init__(self):
        self.voxel_size = 0.05  # 5cm voxel size
        self.distance_threshold = 0.02  # 2cm for plane segmentation

    def load_point_cloud(self, file_path):
        """Load point cloud from file"""
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

    def preprocess_point_cloud(self, point_cloud):
        """Preprocess point cloud with filtering and downsampling"""
        # Remove statistical outliers
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        filtered_cloud = point_cloud.select_by_index(ind)

        # Downsample using voxel grid filter
        downsampled = filtered_cloud.voxel_down_sample(voxel_size=self.voxel_size)

        # Estimate normals for surface analysis
        downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        return downsampled

    def segment_planes(self, point_cloud):
        """Segment planar surfaces (floors, walls, tables) using RANSAC"""
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )

        # Extract plane points and remaining points
        plane_cloud = point_cloud.select_by_index(inliers)
        remaining_cloud = point_cloud.select_by_index(inliers, invert=True)

        return plane_model, plane_cloud, remaining_cloud

    def cluster_objects(self, point_cloud, eps=0.05, min_points=10):
        """Cluster points into separate objects using DBSCAN"""
        points = np.asarray(point_cloud.points)

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
        labels = clustering.labels_

        # Extract clusters
        unique_labels = set(labels)
        clusters = []

        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            # Extract points for this cluster
            cluster_points = points[labels == label]
            cluster_cloud = o3d.geometry.PointCloud()
            cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)

            clusters.append(cluster_cloud)

        return clusters

    def extract_features(self, point_cloud):
        """Extract geometric features from point cloud"""
        points = np.asarray(point_cloud.points)

        # Calculate basic statistics
        centroid = np.mean(points, axis=0)
        bbox = np.ptp(points, axis=0)  # Bounding box dimensions
        volume = np.prod(bbox)

        # Calculate eigenvalues for shape analysis
        cov_matrix = np.cov(points.T)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)

        # Shape descriptors
        linearity = (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2]
        planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
        sphericity = eigenvalues[0] / eigenvalues[2]

        features = {
            'centroid': centroid,
            'bbox': bbox,
            'volume': volume,
            'linearity': linearity,
            'planarity': planarity,
            'sphericity': sphericity,
            'point_count': len(points)
        }

        return features

    def register_point_clouds(self, source_cloud, target_cloud):
        """Register two point clouds using ICP"""
        # Initial alignment using FPFH features
        source_fpfh = self.compute_fpfh_features(source_cloud)
        target_fpfh = self.compute_fpfh_features(target_cloud)

        # Initial registration using RANSAC
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_cloud, target_cloud, source_fpfh, target_fpfh,
            mutual_filter=True,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        # Refine with ICP
        result_icp = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud,
            max_correspondence_distance=0.02,
            init=result_ransac.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        return result_icp

    def compute_fpfh_features(self, point_cloud):
        """Compute Fast Point Feature Histograms"""
        radius_normal = self.voxel_size * 2
        point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        radius_feature = self.voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            point_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )

        return fpfh

# Example usage
def process_environment_point_cloud(pcd_file):
    """Process a point cloud of an environment"""
    processor = PointCloudProcessor()

    # Load and preprocess
    raw_cloud = processor.load_point_cloud(pcd_file)
    processed_cloud = processor.preprocess_point_cloud(raw_cloud)

    # Segment planes (floors, walls)
    plane_model, plane_cloud, object_cloud = processor.segment_planes(processed_cloud)

    # Cluster objects
    object_clusters = processor.cluster_objects(object_cloud)

    print(f"Found {len(object_clusters)} objects in the environment")
    print(f"Plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")

    return object_clusters, plane_cloud
```

## Depth Image Analysis

### Converting Depth Images to Point Clouds

Depth cameras provide 2.5D information that can be converted to 3D point clouds:

```python
class DepthImageProcessor:
    def __init__(self, fx=525.0, fy=525.0, cx=319.5, cy=239.5):
        self.fx = fx  # Focal length x
        self.fy = fy  # Focal length y
        self.cx = cx  # Principal point x
        self.cy = cy  # Principal point y

    def depth_to_point_cloud(self, depth_image, color_image=None):
        """Convert depth image to point cloud"""
        height, width = depth_image.shape

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Convert to 3D coordinates
        z_coords = depth_image.astype(np.float32)
        x_coords = (x_coords - self.cx) * z_coords / self.fx
        y_coords = (y_coords - self.cy) * z_coords / self.fy

        # Stack to get 3D points
        points = np.stack([x_coords, y_coords, z_coords], axis=-1)

        # Reshape to (N, 3)
        points = points.reshape(-1, 3)

        # Remove invalid points (zeros, infinities, negative depths)
        valid_mask = (
            np.isfinite(points).all(axis=1) &
            (points[:, 2] > 0) &
            (points[:, 2] < 10.0)  # Reasonable depth range
        )
        points = points[valid_mask]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if color_image is not None:
            # Add colors to valid points
            colors = color_image.reshape(-1, 3)[valid_mask] / 255.0  # Normalize to [0,1]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def filter_depth_outliers(self, depth_image, kernel_size=5, threshold=3):
        """Filter outliers in depth image using median filtering"""
        # Apply median filter
        median_filtered = cv2.medianBlur(depth_image.astype(np.float32), kernel_size)

        # Calculate difference from median
        diff = np.abs(depth_image.astype(np.float32) - median_filtered)

        # Calculate local standard deviation
        local_mean = cv2.blur(depth_image.astype(np.float32), (kernel_size, kernel_size))
        local_var = cv2.blur((depth_image.astype(np.float32) - local_mean)**2, (kernel_size, kernel_size))
        local_std = np.sqrt(local_var)

        # Create mask for outliers
        outlier_mask = diff > threshold * (local_std + 1e-6)  # Add small epsilon to avoid division by zero

        # Replace outliers with median values
        filtered_depth = depth_image.copy().astype(np.float32)
        filtered_depth[outlier_mask] = median_filtered[outlier_mask]

        return filtered_depth

    def extract_surface_normals(self, depth_image):
        """Extract surface normals from depth image"""
        # Compute gradients
        grad_x = np.gradient(depth_image, axis=1)
        grad_y = np.gradient(depth_image, axis=0)

        # Create normal vectors
        normals = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        normals[:, :, 0] = -grad_x  # x component
        normals[:, :, 1] = -grad_y  # y component
        normals[:, :, 2] = 1.0      # z component (up)

        # Normalize
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / norm

        return normals

    def detect_planes_in_depth(self, depth_image):
        """Detect planar surfaces directly from depth image"""
        # Convert to point cloud first
        pcd = self.depth_to_point_cloud(depth_image)

        # Segment planes using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,  # 1cm threshold
            ransac_n=3,
            num_iterations=1000
        )

        return plane_model, inliers
```

## SLAM Fundamentals

### Visual SLAM Pipeline

```python
import cv2
import numpy as np
from collections import deque

class VisualSLAM:
    def __init__(self):
        # Feature detection and matching
        self.detector = cv2.SIFT_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher()

        # Pose estimation
        self.current_pose = np.eye(4)  # 4x4 homogeneous transformation
        self.keyframes = []
        self.map_points = []

        # Tracking
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None

        # Trajectory
        self.trajectory = []

    def process_frame(self, image, camera_matrix):
        """Process a new camera frame for SLAM"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if self.prev_frame is None:
            # First frame - initialize
            self.prev_frame = gray
            self.prev_kp = keypoints
            self.prev_desc = descriptors
            self.keyframes.append({
                'image': image,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'pose': self.current_pose.copy()
            })
            return self.current_pose

        if descriptors is None or self.prev_desc is None:
            return self.current_pose

        # Match features with previous frame
        matches = self.matcher.knnMatch(descriptors, self.prev_desc, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            # Not enough matches to estimate motion
            return self.current_pose

        # Extract matched points
        curr_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        prev_pts = np.float32([self.prev_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate motion using Essential Matrix
        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is not None:
            # Recover pose
            _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, camera_matrix)

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            # Update current pose
            self.current_pose = self.current_pose @ np.linalg.inv(T)

        # Store current frame data
        self.prev_frame = gray
        self.prev_kp = keypoints
        self.prev_desc = descriptors

        # Store trajectory
        self.trajectory.append(self.current_pose[:3, 3].copy())

        # Add keyframe if significant motion occurred
        if self.should_add_keyframe():
            self.add_keyframe(image, keypoints, descriptors, self.current_pose)

        return self.current_pose

    def should_add_keyframe(self):
        """Determine if current frame should be a keyframe"""
        if not self.trajectory:
            return True

        # Add keyframe if moved significantly
        current_pos = self.trajectory[-1]
        last_keyframe_pos = self.keyframes[-1]['pose'][:3, 3]

        distance = np.linalg.norm(current_pos - last_keyframe_pos)

        return distance > 0.5  # Add keyframe every 50cm

    def add_keyframe(self, image, keypoints, descriptors, pose):
        """Add current frame as a keyframe"""
        keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy()
        }
        self.keyframes.append(keyframe)

    def get_trajectory(self):
        """Return the robot trajectory"""
        return np.array(self.trajectory)

class OccupancyGridMapper:
    def __init__(self, resolution=0.1, grid_size=100):
        self.resolution = resolution
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))  # Log odds representation
        self.origin = np.array([grid_size//2, grid_size//2])  # Center of grid

    def update_with_lidar_scan(self, robot_pose, ranges, angles):
        """Update occupancy grid with LiDAR scan"""
        # Convert robot pose to grid coordinates
        robot_x, robot_y, robot_theta = robot_pose

        # Robot position in grid coordinates
        robot_grid_x = int(robot_x / self.resolution) + self.origin[0]
        robot_grid_y = int(robot_y / self.resolution) + self.origin[1]

        # Update grid for each laser beam
        for i, (range_val, angle) in enumerate(zip(ranges, angles)):
            if range_val < 0.1 or range_val > 10.0:  # Invalid range
                continue

            # Calculate endpoint of laser beam in world coordinates
            world_end_x = robot_x + range_val * np.cos(robot_theta + angle)
            world_end_y = robot_y + range_val * np.sin(robot_theta + angle)

            # Convert to grid coordinates
            grid_end_x = int(world_end_x / self.resolution) + self.origin[0]
            grid_end_y = int(world_end_y / self.resolution) + self.origin[1]

            # Bresenham's algorithm to trace the beam
            self.trace_beam(robot_grid_x, robot_grid_y, grid_end_x, grid_end_y, hit=(i==len(ranges)-1))

    def trace_beam(self, x0, y0, x1, y1, hit=True):
        """Trace a beam from (x0,y0) to (x1,y1) and update occupancy"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        # Update cells along the beam (free space)
        while x != x1 or y != y1:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Update with free space assumption
                self.grid[y, x] -= 0.4  # Decrease occupancy (log odds)

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        # Update endpoint (occupied space if hit)
        if hit and 0 <= x1 < self.grid_size and 0 <= y1 < self.grid_size:
            self.grid[y1, x1] += 0.6  # Increase occupancy (log odds)

    def get_probability_grid(self):
        """Convert log odds to probability"""
        prob_grid = 1 - (1 / (1 + np.exp(self.grid)))
        return prob_grid

    def visualize_grid(self):
        """Visualize the occupancy grid"""
        prob_grid = self.get_probability_grid()
        plt.imshow(prob_grid, cmap='gray', origin='lower')
        plt.colorbar(label='Occupancy Probability')
        plt.title('Occupancy Grid Map')
        plt.show()
```

## 3D Object Detection and Recognition

### 3D Object Detection Pipeline

```python
class ThreeDObjectDetector:
    def __init__(self):
        self.point_cloud_processor = PointCloudProcessor()
        self.feature_extractor = ThreeDFeatureExtractor()
        self.object_detector = ThreeDObjectDetectorModel()

    def detect_objects_in_point_cloud(self, point_cloud):
        """Detect 3D objects in point cloud"""
        # Preprocess point cloud
        processed_cloud = self.point_cloud_processor.preprocess_point_cloud(point_cloud)

        # Segment planes to isolate objects
        _, _, object_cloud = self.point_cloud_processor.segment_planes(processed_cloud)

        # Cluster objects
        clusters = self.point_cloud_processor.cluster_objects(object_cloud)

        # Extract features for each cluster
        detected_objects = []
        for i, cluster in enumerate(clusters):
            # Extract features
            features = self.feature_extractor.extract_features(cluster)

            # Classify object
            obj_class, confidence = self.object_detector.classify_object(features)

            # Estimate bounding box
            bbox = self.estimate_bounding_box(cluster)

            detected_objects.append({
                'id': i,
                'class': obj_class,
                'confidence': confidence,
                'bbox': bbox,
                'centroid': features['centroid'],
                'cluster': cluster
            })

        return detected_objects

    def estimate_bounding_box(self, point_cloud):
        """Estimate oriented bounding box for point cloud"""
        points = np.asarray(point_cloud.points)

        # Compute bounding box
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        center = (min_pt + max_pt) / 2.0
        size = max_pt - min_pt

        return {
            'center': center,
            'size': size,
            'min_point': min_pt,
            'max_point': max_pt
        }

class ThreeDFeatureExtractor:
    def extract_geometric_features(self, point_cloud):
        """Extract geometric features from point cloud"""
        points = np.asarray(point_cloud.points)

        # Statistical features
        centroid = np.mean(points, axis=0)
        variance = np.var(points, axis=0)
        spread = np.ptp(points, axis=0)

        # Shape features
        cov_matrix = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Shape descriptors
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        sphericity = eigenvalues[2] / eigenvalues[0]

        return {
            'centroid': centroid,
            'variance': variance,
            'spread': spread,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'linearity': linearity,
            'planarity': planarity,
            'sphericity': sphericity
        }

    def extract_local_features(self, point_cloud, radius=0.1):
        """Extract local geometric features"""
        points = np.asarray(point_cloud.points)

        # Compute local features for each point
        local_features = []

        for i, pt in enumerate(points):
            # Find neighbors within radius
            distances = np.linalg.norm(points - pt, axis=1)
            neighbors = points[distances < radius]

            if len(neighbors) < 3:
                continue

            # Compute local covariance
            cov = np.cov(neighbors.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)

            # Sort eigenvalues
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]

            # Local features
            linearity = max(0, (eigenvals[0] - eigenvals[1]) / eigenvals[0])
            planarity = max(0, (eigenvals[1] - eigenvals[2]) / eigenvals[0])
            scattering = eigenvals[2] / eigenvals[0]
            verticality = abs(eigenvecs[2, 2])  # How vertical the normal is

            local_features.append({
                'point': pt,
                'linearity': linearity,
                'planarity': planarity,
                'scattering': scattering,
                'verticality': verticality
            })

        return local_features
```

## Environment Understanding

### Scene Graph Construction

```python
class SceneGraphBuilder:
    def __init__(self):
        self.objects = []
        self.spatial_relations = []
        self.temporal_relations = []

    def build_scene_graph(self, detected_objects, robot_pose):
        """Build scene graph from detected objects and robot pose"""
        # Create nodes for objects
        nodes = []
        for obj in detected_objects:
            node = {
                'id': obj['id'],
                'type': obj['class'],
                'position': obj['centroid'],
                'bbox': obj['bbox'],
                'confidence': obj['confidence']
            }
            nodes.append(node)

        # Add robot node
        robot_node = {
            'id': 'robot',
            'type': 'robot',
            'position': robot_pose[:3],
            'bbox': None,
            'confidence': 1.0
        }
        nodes.append(robot_node)

        # Compute spatial relations
        relations = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    # Calculate spatial relation
                    rel_type, strength = self.compute_spatial_relation(
                        node1['position'], node2['position']
                    )

                    if strength > 0.1:  # Significant relation
                        relations.append({
                            'subject': node1['id'],
                            'predicate': rel_type,
                            'object': node2['id'],
                            'strength': strength
                        })

        scene_graph = {
            'nodes': nodes,
            'relations': relations,
            'timestamp': time.time()
        }

        return scene_graph

    def compute_spatial_relation(self, pos1, pos2):
        """Compute spatial relation between two positions"""
        diff = pos2 - pos1
        distance = np.linalg.norm(diff)

        # Determine dominant direction
        if abs(diff[0]) > max(abs(diff[1]), abs(diff[2])):
            if diff[0] > 0:
                relation = 'right_of'
            else:
                relation = 'left_of'
        elif abs(diff[1]) > abs(diff[2]):
            if diff[1] > 0:
                relation = 'above'
            else:
                relation = 'below'
        else:
            if diff[2] > 0:
                relation = 'behind'
            else:
                relation = 'in_front_of'

        # Relation strength decreases with distance
        strength = max(0.0, 1.0 - distance/5.0)  # Strong up to 5m

        return relation, strength

class EnvironmentRepresentation:
    def __init__(self):
        self.topological_map = {}  # Connectivity between locations
        self.semantic_map = {}     # Object locations and attributes
        self.metric_map = None     # Detailed geometric map

    def update_environment(self, sensor_data, robot_pose):
        """Update environment representation with new sensor data"""
        # Process 3D sensor data
        if 'point_cloud' in sensor_data:
            self.update_metric_map(sensor_data['point_cloud'], robot_pose)

        if 'detected_objects' in sensor_data:
            self.update_semantic_map(sensor_data['detected_objects'], robot_pose)

        # Update topological connectivity
        self.update_topological_map(robot_pose)

    def update_metric_map(self, point_cloud, robot_pose):
        """Update detailed metric map"""
        # Integrate point cloud into global map
        # This would typically use scan-to-map registration
        pass

    def update_semantic_map(self, objects, robot_pose):
        """Update semantic map with object locations"""
        for obj in objects:
            obj_id = obj['id']
            world_pos = self.transform_to_world_frame(obj['centroid'], robot_pose)

            self.semantic_map[obj_id] = {
                'class': obj['class'],
                'position': world_pos,
                'confidence': obj['confidence'],
                'last_seen': time.time()
            }

    def transform_to_world_frame(self, local_pos, robot_pose):
        """Transform local position to world frame"""
        # Apply robot pose transformation
        R = self.rotation_matrix_from_pose(robot_pose)
        world_pos = R @ local_pos + robot_pose[:3]
        return world_pos

    def rotation_matrix_from_pose(self, pose):
        """Extract rotation matrix from pose"""
        # If pose is 6DOF [x,y,z,r,p,yaw], convert to rotation matrix
        # If pose is already 4x4 matrix, extract rotation part
        if pose.shape == (6,):
            x, y, z, roll, pitch, yaw = pose
            # Convert Euler angles to rotation matrix
            R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        elif pose.shape == (4, 4):
            R = pose[:3, :3]
        else:
            raise ValueError("Invalid pose format")

        return R

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles to rotation matrix"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        return R
```

## Practical Applications

### Mobile Robot Navigation

```python
class NavigationSystem:
    def __init__(self):
        self.slam_system = VisualSLAM()
        self.mapper = OccupancyGridMapper()
        self.obstacle_detector = ThreeDObjectDetector()
        self.path_planner = PathPlanner()

    def navigate(self, goal_position, sensor_data, camera_matrix):
        """Navigate to goal position using 3D perception"""
        # Update SLAM with current frame
        current_pose = self.slam_system.process_frame(
            sensor_data['image'], camera_matrix
        )

        # Process point cloud for obstacles
        if 'point_cloud' in sensor_data:
            obstacles = self.obstacle_detector.detect_objects_in_point_cloud(
                sensor_data['point_cloud']
            )

            # Update occupancy grid with obstacle information
            for obstacle in obstacles:
                self.update_occupancy_for_obstacle(obstacle, current_pose)

        # Plan path to goal
        path = self.path_planner.plan_path(
            current_pose[:3, 3],  # Current position
            goal_position,
            self.mapper.get_probability_grid()
        )

        return path, current_pose

    def update_occupancy_for_obstacle(self, obstacle, robot_pose):
        """Update occupancy grid based on detected obstacle"""
        # Transform obstacle position to robot frame
        obstacle_world_pos = obstacle['centroid']
        obstacle_robot_frame = self.transform_to_robot_frame(
            obstacle_world_pos, robot_pose
        )

        # Update grid
        grid_x = int(obstacle_robot_frame[0] / self.mapper.resolution) + self.mapper.origin[0]
        grid_y = int(obstacle_robot_frame[1] / self.mapper.resolution) + self.mapper.origin[1]

        if (0 <= grid_x < self.mapper.grid_size and
            0 <= grid_y < self.mapper.grid_size):
            # Mark as occupied
            self.mapper.grid[grid_y, grid_x] += 1.0

class PathPlanner:
    def __init__(self):
        self.grid_resolution = 0.1  # 10cm resolution

    def plan_path(self, start_pos, goal_pos, occupancy_grid):
        """Plan path using A* algorithm on occupancy grid"""
        import heapq

        # Convert positions to grid coordinates
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)

        # A* algorithm
        heap = [(0, start_grid)]  # (cost, position)
        came_from = {}
        cost_so_far = {}
        came_from[tuple(start_grid)] = None
        cost_so_far[tuple(start_grid)] = 0

        while heap:
            current_cost, current = heapq.heappop(heap)

            if current[0] == goal_grid[0] and current[1] == goal_grid[1]:
                break

            # Check 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = [current[0] + dx, current[1] + dy]

                    if (0 <= neighbor[0] < occupancy_grid.shape[0] and
                        0 <= neighbor[1] < occupancy_grid.shape[1]):

                        # Skip if occupied
                        if occupancy_grid[neighbor[0], neighbor[1]] > 0.7:
                            continue

                        # Calculate movement cost
                        move_cost = np.sqrt(dx*dx + dy*dy)
                        new_cost = cost_so_far[tuple(current)] + move_cost

                        if (tuple(neighbor) not in cost_so_far or
                            new_cost < cost_so_far[tuple(neighbor)]):

                            cost_so_far[tuple(neighbor)] = new_cost
                            priority = new_cost + self.heuristic(neighbor, goal_grid)
                            heapq.heappush(heap, (priority, neighbor))
                            came_from[tuple(neighbor)] = current

        # Reconstruct path
        path = []
        current = tuple(goal_grid)
        while current in came_from and came_from[current] is not None:
            path.append(self.grid_to_world(np.array(current)))
            current = tuple(came_from[current])

        path.reverse()  # From start to goal
        return path

    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(world_pos[0] / self.grid_resolution)
        grid_y = int(world_pos[1] / self.grid_resolution)
        return np.array([grid_x, grid_y])

    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_pos[0] * self.grid_resolution
        world_y = grid_pos[1] * self.grid_resolution
        return np.array([world_x, world_y, 0.0])  # Z=0 for 2D navigation

    def heuristic(self, pos1, pos2):
        """Heuristic function for A* (Euclidean distance)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```

## Learning Objectives

By the end of this lesson, you should be able to:
- Process and analyze point cloud data for 3D environment understanding
- Convert depth images to 3D point clouds and extract meaningful features
- Implement SLAM algorithms for simultaneous localization and mapping
- Detect and recognize 3D objects from sensor data
- Build semantic representations of environments from 3D perception
- Apply 3D perception techniques to mobile robot navigation
- Understand the relationship between 3D perception and robot autonomy
- Evaluate 3D perception system performance and accuracy
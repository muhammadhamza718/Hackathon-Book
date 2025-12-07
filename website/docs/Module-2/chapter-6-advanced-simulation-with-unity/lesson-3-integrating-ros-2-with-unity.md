---
title: 'Integrating ROS 2 with Unity'
description: 'Integrating ROS 2 with Unity simulations, describing methods for connecting ROS 2 with Unity, data exchange, controlling simulated robots, receiving sensor data'
chapter: 6
lesson: 3
module: 2
sidebar_label: 'Integrating ROS 2 with Unity'
sidebar_position: 3
tags: ['ROS 2', 'Unity', 'Integration', 'Data Exchange', 'Robot Control', 'Sensor Data']
keywords: ['ROS 2', 'Unity', 'integration', 'data exchange', 'robot control', 'sensor data', 'TCP communication']
---

# Integrating ROS 2 with Unity

## Overview

Integrating ROS 2 with Unity enables powerful simulation capabilities that combine Unity's high-fidelity graphics and physics with ROS 2's robotics middleware. This integration allows for photorealistic simulation, advanced sensor modeling, and realistic environment rendering while maintaining compatibility with the ROS 2 ecosystem. This lesson covers various methods for connecting ROS 2 with Unity, implementing data exchange mechanisms, and controlling simulated robots while receiving sensor data.

## Integration Architecture

### Communication Methods

Unity can integrate with ROS 2 through several approaches:

1. **TCP/IP Bridge**: Direct TCP communication between Unity and ROS 2 nodes
2. **ROS-TCP-Connector**: Unity package for seamless ROS communication
3. **WebSocket Communication**: For web-based Unity deployments
4. **Custom Bridge Nodes**: Specialized ROS nodes that handle Unity communication

### System Architecture

The typical integration architecture includes:

```
ROS 2 Nodes ←→ Bridge ←→ Unity Simulation
    ↑              ↑           ↑
Messages ←→ TCP/IP ←→ Unity ←→ Physics/Sensors
```

## ROS-TCP-Connector Setup

### Installation and Configuration

The ROS-TCP-Connector package provides the easiest integration method:

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class UnityROSBridge : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public bool useManualIp = false;

    private ROSConnection ros;

    void Start()
    {
        SetupROSConnection();
    }

    void SetupROSConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();

        if (useManualIp)
        {
            ros.Initialize(rosIPAddress, rosPort);
        }
        else
        {
            // Use default localhost
            ros.Initialize();
        }

        // Register publishers and subscribers
        ros.RegisterPublisher<TwistMsg>("/cmd_vel");
        ros.RegisterSubscriber<LaserScanMsg>("/scan", OnLaserScanReceived);
        ros.RegisterSubscriber<OdometryMsg>("/odom", OnOdometryReceived);
    }

    // Example of sending robot commands
    public void SendVelocityCommand(float linearX, float angularZ)
    {
        var twist = new TwistMsg();
        twist.linear = new Vector3Msg(linearX, 0, 0);
        twist.angular = new Vector3Msg(0, 0, angularZ);

        ros.Publish("/cmd_vel", twist);
    }

    // Callback for receiving laser scan data
    void OnLaserScanReceived(LaserScanMsg scan)
    {
        // Process laser scan data
        Debug.Log($"Received laser scan with {scan.ranges.Length} points");

        // Update Unity visualization or process data
        ProcessLaserScan(scan);
    }

    // Callback for receiving odometry data
    void OnOdometryReceived(OdometryMsg odom)
    {
        // Process odometry data
        Debug.Log($"Received odometry: pos=({odom.pose.pose.position.x}, {odom.pose.pose.position.y})");

        // Update Unity robot position or process data
        ProcessOdometry(odom);
    }

    void ProcessLaserScan(LaserScanMsg scan)
    {
        // Example: Visualize laser scan in Unity
        // This could update a debug visualization or sensor display
    }

    void ProcessOdometry(OdometryMsg odom)
    {
        // Example: Update Unity robot position based on odometry
        // This could be used for visualization or comparison with ground truth
    }
}
```

### Message Type Support

Unity supports various ROS message types:

```csharp
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

public class MessageHandler : MonoBehaviour
{
    // Publisher examples
    private MessageExtensionsPublisher<ImageMsg> imagePublisher;
    private MessageExtensionsPublisher<JointStateMsg> jointStatePublisher;
    private MessageExtensionsPublisher<ImuMsg> imuPublisher;

    void Start()
    {
        SetupPublishers();
    }

    void SetupPublishers()
    {
        // Image publisher
        imagePublisher = new MessageExtensionsPublisher<ImageMsg>(
            ROSConnection.GetOrCreateInstance(), "/camera/image_raw");

        // Joint state publisher
        jointStatePublisher = new MessageExtensionsPublisher<JointStateMsg>(
            ROSConnection.GetOrCreateInstance(), "/joint_states");

        // IMU publisher
        imuPublisher = new MessageExtensionsPublisher<ImuMsg>(
            ROSConnection.GetOrCreateInstance(), "/imu/data");
    }

    // Publish image data
    public void PublishImage(Texture2D imageTexture, string frameId = "camera_frame")
    {
        ImageMsg imageMsg = new ImageMsg();

        // Set image properties
        imageMsg.header = new HeaderMsg();
        imageMsg.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        imageMsg.header.frame_id = frameId;

        imageMsg.height = (uint)imageTexture.height;
        imageMsg.width = (uint)imageTexture.width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(imageTexture.width * 3); // 3 bytes per pixel for RGB

        // Convert texture to byte array
        Color32[] colors = imageTexture.GetPixels32();
        byte[] imageData = new byte[colors.Length * 3];

        for (int i = 0; i < colors.Length; i++)
        {
            imageData[i * 3] = colors[i].r;
            imageData[i * 3 + 1] = colors[i].g;
            imageData[i * 3 + 2] = colors[i].b;
        }

        imageMsg.data = imageData;

        imagePublisher.Publish(imageMsg);
    }

    // Publish joint states
    public void PublishJointStates(string[] jointNames, float[] positions, float[] velocities, float[] efforts)
    {
        JointStateMsg jointStateMsg = new JointStateMsg();

        jointStateMsg.header = new HeaderMsg();
        jointStateMsg.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        jointStateMsg.header.frame_id = "base_link";

        jointStateMsg.name = jointNames;
        jointStateMsg.position = positions;
        jointStateMsg.velocity = velocities;
        jointStateMsg.effort = efforts;

        jointStatePublisher.Publish(jointStateMsg);
    }

    // Publish IMU data
    public void PublishIMUData(Vector3 linearAccel, Vector3 angularVel, Vector4 orientation)
    {
        ImuMsg imuMsg = new ImuMsg();

        imuMsg.header = new HeaderMsg();
        imuMsg.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        imuMsg.header.frame_id = "imu_link";

        imuMsg.linear_acceleration = new Vector3Msg(linearAccel.x, linearAccel.y, linearAccel.z);
        imuMsg.angular_velocity = new Vector3Msg(angularVel.x, angularVel.y, angularVel.z);
        imuMsg.orientation = new QuaternionMsg(orientation.x, orientation.y, orientation.z, orientation.w);

        // Set covariance matrices to -1 if not available
        for (int i = 0; i < 9; i++)
        {
            imuMsg.linear_acceleration_covariance[i] = -1.0f;
            imuMsg.angular_velocity_covariance[i] = -1.0f;
            imuMsg.orientation_covariance[i] = -1.0f;
        }

        imuPublisher.Publish(imuMsg);
    }
}
```

## Robot Control Systems

### Velocity Control Interface

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    [Header("Robot Properties")]
    public float maxLinearVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;
    public float wheelBase = 0.5f; // Distance between wheels for diff drive
    public float wheelRadius = 0.05f;

    [Header("Motor Settings")]
    public float motorTorque = 10.0f;
    public float motorSpeed = 100.0f;

    private ROSConnection ros;
    private Rigidbody rb;
    private Vector3 targetVelocity = Vector3.zero;
    private float targetAngularVelocity = 0.0f;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        rb = GetComponent<Rigidbody>();

        // Register for velocity commands
        ros.RegisterSubscriber<TwistMsg>("/cmd_vel", OnVelocityCommandReceived);
    }

    void Update()
    {
        // Apply velocity control
        ApplyVelocityControl();
    }

    void OnVelocityCommandReceived(TwistMsg twist)
    {
        // Convert ROS velocity commands to Unity velocities
        targetVelocity = new Vector3(twist.linear.x, 0, twist.linear.y);
        targetAngularVelocity = twist.angular.z;

        // Clamp velocities to maximum values
        targetVelocity = Vector3.ClampMagnitude(targetVelocity, maxLinearVelocity);
        targetAngularVelocity = Mathf.Clamp(targetAngularVelocity, -maxAngularVelocity, maxAngularVelocity);

        Debug.Log($"Received velocity command: linear=({twist.linear.x}, {twist.linear.y}), angular={twist.angular.z}");
    }

    void ApplyVelocityControl()
    {
        if (rb != null)
        {
            // Apply linear velocity
            Vector3 linearForce = (targetVelocity - rb.velocity) * motorTorque;
            rb.AddForce(linearForce, ForceMode.Force);

            // Apply angular velocity (torque around Y-axis for rotation)
            float angularDiff = targetAngularVelocity - rb.angularVelocity.y;
            float angularTorque = angularDiff * motorTorque;
            rb.AddTorque(Vector3.up * angularTorque, ForceMode.Force);
        }
    }

    // Alternative: Differential drive control
    public void ApplyDifferentialDriveControl(float leftVelocity, float rightVelocity)
    {
        if (rb != null)
        {
            // Calculate linear and angular velocities from wheel velocities
            float linearVel = (leftVelocity + rightVelocity) / 2.0f;
            float angularVel = (rightVelocity - leftVelocity) / wheelBase;

            // Apply forces
            Vector3 linearForce = transform.forward * linearVel * motorTorque;
            float angularTorque = angularVel * motorTorque;

            rb.AddForce(linearForce, ForceMode.Force);
            rb.AddTorque(Vector3.up * angularTorque, ForceMode.Force);
        }
    }

    // Get current robot state for publishing
    public OdometryMsg GetOdometry()
    {
        OdometryMsg odom = new OdometryMsg();

        odom.header = new HeaderMsg();
        odom.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";

        // Position
        odom.pose.pose.position = new Vector3Msg(transform.position.x, transform.position.y, transform.position.z);

        // Convert Unity rotation (Quaternion) to ROS format
        Quaternion unityRot = transform.rotation;
        odom.pose.pose.orientation = new QuaternionMsg(unityRot.x, unityRot.y, unityRot.z, unityRot.w);

        // Velocity
        odom.twist.twist.linear = new Vector3Msg(rb.velocity.x, rb.velocity.y, rb.velocity.z);
        odom.twist.twist.angular = new Vector3Msg(rb.angularVelocity.x, rb.angularVelocity.y, rb.angularVelocity.z);

        return odom;
    }
}
```

### Joint Control Interface

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Control;
using RosMessageTypes.Sensor;

public class UnityJointController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public string[] jointNames;
    public ArticulationBody[] jointArticulationBodies;
    public float[] jointPositions;
    public float[] jointVelocities;
    public float[] jointEfforts;

    private ROSConnection ros;
    private float[] targetPositions;
    private float[] targetVelocities;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Initialize arrays
        jointPositions = new float[jointArticulationBodies.Length];
        jointVelocities = new float[jointArticulationBodies.Length];
        jointEfforts = new float[jointArticulationBodies.Length];
        targetPositions = new float[jointArticulationBodies.Length];
        targetVelocities = new float[jointArticulationBodies.Length];

        // Register for joint commands
        ros.RegisterSubscriber<JointTrajectoryMsg>("/joint_trajectory", OnJointTrajectoryReceived);
        ros.RegisterSubscriber<FollowJointTrajectoryActionGoal>("/follow_joint_trajectory/goal", OnFollowJointTrajectoryGoalReceived);
    }

    void Update()
    {
        // Update joint states
        UpdateJointStates();

        // Apply joint control
        ApplyJointControl();
    }

    void UpdateJointStates()
    {
        for (int i = 0; i < jointArticulationBodies.Length; i++)
        {
            if (jointArticulationBodies[i] != null)
            {
                jointPositions[i] = jointArticulationBodies[i].jointPosition[0];
                jointVelocities[i] = jointArticulationBodies[i].jointVelocity[0];
                jointEfforts[i] = jointArticulationBodies[i].jointForce[0];
            }
        }
    }

    void ApplyJointControl()
    {
        for (int i = 0; i < jointArticulationBodies.Length; i++)
        {
            if (jointArticulationBodies[i] != null)
            {
                // Set target position
                ArticulationDrive drive = jointArticulationBodies[i].xDrive;
                drive.target = targetPositions[i];
                drive.targetVelocity = targetVelocities[i];
                drive.forceLimit = 1000f; // Maximum force
                drive.damping = 10f; // Damping for smooth movement
                drive.stiffness = 10000f; // Stiffness for accurate positioning
                jointArticulationBodies[i].xDrive = drive;
            }
        }
    }

    void OnJointTrajectoryReceived(JointTrajectoryMsg trajectory)
    {
        if (trajectory.points.Length > 0)
        {
            // Get the first point (or interpolate based on time)
            var point = trajectory.points[0];

            for (int i = 0; i < jointNames.Length; i++)
            {
                // Find the index of this joint in the trajectory message
                int jointIndex = System.Array.IndexOf(trajectory.joint_names, jointNames[i]);
                if (jointIndex >= 0 && jointIndex < point.positions.Length)
                {
                    targetPositions[i] = (float)point.positions[jointIndex];

                    if (point.velocities != null && jointIndex < point.velocities.Length)
                    {
                        targetVelocities[i] = (float)point.velocities[jointIndex];
                    }
                }
            }
        }
    }

    void OnFollowJointTrajectoryGoalReceived(FollowJointTrajectoryActionGoal goal)
    {
        // Handle action-based trajectory following
        OnJointTrajectoryReceived(goal.goal.trajectory);
    }

    // Publish joint states
    public JointStateMsg GetJointStates()
    {
        JointStateMsg jointState = new JointStateMsg();

        jointState.header = new HeaderMsg();
        jointState.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        jointState.header.frame_id = "base_link";

        jointState.name = jointNames;
        jointState.position = System.Array.ConvertAll(jointPositions, x => (double)x);
        jointState.velocity = System.Array.ConvertAll(jointVelocities, x => (double)x);
        jointState.effort = System.Array.ConvertAll(jointEfforts, x => (double)x);

        return jointState;
    }

    // Inverse kinematics example
    public void MoveToEndEffector(Vector3 targetPosition, Quaternion targetRotation)
    {
        // This would implement inverse kinematics to calculate joint angles
        // For now, we'll just move the end effector directly
        // In a real implementation, you'd use Unity's built-in IK or custom algorithms

        // Calculate joint angles to reach target position
        // This is a simplified example
        Debug.Log($"Moving end effector to: {targetPosition}, rotation: {targetRotation}");
    }
}
```

## Sensor Data Integration

### Camera Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera sensorCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float fieldOfView = 60.0f;
    public string frameId = "camera_frame";
    public float minRange = 0.1f;
    public float maxRange = 10.0f;

    [Header("Sensor Settings")]
    public string rosTopic = "/camera/image_raw";
    public float publishRate = 30.0f; // Hz

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private float lastPublishTime = 0.0f;
    private Texture2D tempTexture;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        SetupCamera();
        SetupRenderTexture();

        // Create temporary texture for reading pixels
        tempTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }

    void SetupCamera()
    {
        sensorCamera.fieldOfView = fieldOfView;
        sensorCamera.aspect = (float)imageWidth / imageHeight;
    }

    void SetupRenderTexture()
    {
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        sensorCamera.targetTexture = renderTexture;
    }

    void Update()
    {
        float currentTime = Time.time;
        if (currentTime - lastPublishTime >= 1.0f / publishRate)
        {
            PublishCameraData();
            lastPublishTime = currentTime;
        }
    }

    void PublishCameraData()
    {
        // Capture image from camera
        RenderTexture.active = renderTexture;
        tempTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tempTexture.Apply();
        RenderTexture.active = null;

        // Convert to ROS image message
        ImageMsg imageMsg = CreateImageMessage(tempTexture);

        // Publish to ROS
        ros.Publish(rosTopic, imageMsg);
    }

    ImageMsg CreateImageMessage(Texture2D texture)
    {
        ImageMsg imageMsg = new ImageMsg();

        imageMsg.header = new HeaderMsg();
        imageMsg.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        imageMsg.header.frame_id = frameId;

        imageMsg.height = (uint)texture.height;
        imageMsg.width = (uint)texture.width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(texture.width * 3); // 3 bytes per pixel for RGB

        // Convert texture to byte array
        Color32[] colors = texture.GetPixels32();
        byte[] imageData = new byte[colors.Length * 3];

        for (int i = 0; i < colors.Length; i++)
        {
            // Convert from Unity's RGBA to ROS RGB format
            imageData[i * 3] = colors[i].r;     // R
            imageData[i * 3 + 1] = colors[i].g; // G
            imageData[i * 3 + 2] = colors[i].b; // B
        }

        imageMsg.data = imageData;

        return imageMsg;
    }

    // Publish camera info
    public CameraInfoMsg GetCameraInfo()
    {
        CameraInfoMsg cameraInfo = new CameraInfoMsg();

        cameraInfo.header = new HeaderMsg();
        cameraInfo.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        cameraInfo.header.frame_id = frameId;

        cameraInfo.height = (uint)imageHeight;
        cameraInfo.width = (uint)imageWidth;

        // Intrinsic camera matrix (example values)
        cameraInfo.K = new double[9] {
            525.0, 0.0, imageWidth / 2.0,  // fx, 0, cx
            0.0, 525.0, imageHeight / 2.0, // 0, fy, cy
            0.0, 0.0, 1.0                 // 0, 0, 1
        };

        // Distortion coefficients (assuming no distortion)
        cameraInfo.D = new double[5] { 0.0, 0.0, 0.0, 0.0, 0.0 };

        // Rectification matrix (identity)
        cameraInfo.R = new double[9] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };

        // Projection/camera matrix
        cameraInfo.P = new double[12] {
            525.0, 0.0, 320.0, 0.0,  // fx, 0, cx, Tx
            0.0, 525.0, 240.0, 0.0,  // 0, fy, cy, Ty
            0.0, 0.0, 1.0, 0.0       // 0, 0, 1, Tz
        };

        return cameraInfo;
    }
}
```

### LiDAR Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityLidarSensor : MonoBehaviour
{
    [Header("LiDAR Settings")]
    public int horizontalSamples = 360;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float angleIncrement = 0.01745f; // ~1 degree
    public float minRange = 0.1f;
    public float maxRange = 30.0f;
    public string frameId = "lidar_frame";
    public string rosTopic = "/scan";
    public float publishRate = 10.0f; // Hz

    [Header("Raycast Settings")]
    public LayerMask detectionLayers = -1;
    public float raycastDistance = 30.0f;

    private ROSConnection ros;
    private float lastPublishTime = 0.0f;
    private float[] ranges;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Initialize ranges array
        int numPoints = Mathf.CeilToInt((maxAngle - minAngle) / angleIncrement);
        ranges = new float[numPoints];
    }

    void Update()
    {
        float currentTime = Time.time;
        if (currentTime - lastPublishTime >= 1.0f / publishRate)
        {
            UpdateLidarScan();
            PublishLidarData();
            lastPublishTime = currentTime;
        }
    }

    void UpdateLidarScan()
    {
        // Perform raycasts to simulate LiDAR
        for (int i = 0; i < ranges.Length; i++)
        {
            float angle = minAngle + i * angleIncrement;

            // Calculate ray direction in world space
            Vector3 rayDirection = new Vector3(
                Mathf.Cos(angle),
                0,
                Mathf.Sin(angle)
            );

            // Rotate ray direction based on sensor orientation
            rayDirection = transform.TransformDirection(rayDirection);

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, rayDirection, out hit, raycastDistance, detectionLayers))
            {
                ranges[i] = hit.distance;

                // Apply range limits
                if (ranges[i] < minRange) ranges[i] = minRange;
                if (ranges[i] > maxRange) ranges[i] = Mathf.Infinity; // Use infinity for max range
            }
            else
            {
                ranges[i] = Mathf.Infinity; // No obstacle detected
            }
        }
    }

    void PublishLidarData()
    {
        LaserScanMsg scanMsg = CreateLaserScanMessage();
        ros.Publish(rosTopic, scanMsg);
    }

    LaserScanMsg CreateLaserScanMessage()
    {
        LaserScanMsg scanMsg = new LaserScanMsg();

        scanMsg.header = new HeaderMsg();
        scanMsg.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        scanMsg.header.frame_id = frameId;

        scanMsg.angle_min = minAngle;
        scanMsg.angle_max = maxAngle;
        scanMsg.angle_increment = angleIncrement;
        scanMsg.time_increment = 0.0f; // Not applicable for simulated data
        scanMsg.scan_time = 1.0f / publishRate;
        scanMsg.range_min = minRange;
        scanMsg.range_max = maxRange;

        // Convert float array to double array for ROS message
        scanMsg.ranges = System.Array.ConvertAll(ranges, x => (double)x);

        // Initialize intensities array (optional, can be empty)
        scanMsg.intensities = new double[ranges.Length];
        for (int i = 0; i < scanMsg.intensities.Length; i++)
        {
            scanMsg.intensities[i] = 100.0; // Default intensity value
        }

        return scanMsg;
    }

    // Visualization for debugging
    void OnDrawGizmosSelected()
    {
        if (ranges != null)
        {
            for (int i = 0; i < ranges.Length; i++)
            {
                float angle = minAngle + i * angleIncrement;

                Vector3 rayDirection = new Vector3(
                    Mathf.Cos(angle),
                    0,
                    Mathf.Sin(angle)
                );

                rayDirection = transform.TransformDirection(rayDirection);

                float distance = ranges[i];
                if (float.IsPositiveInfinity(distance))
                {
                    distance = maxRange;
                }

                Vector3 endPos = transform.position + rayDirection * distance;

                Gizmos.color = distance < maxRange ? Color.red : Color.green;
                Gizmos.DrawLine(transform.position, endPos);
            }
        }
    }
}
```

### IMU Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityIMUSensor : MonoBehaviour
{
    [Header("IMU Settings")]
    public string frameId = "imu_link";
    public string rosTopic = "/imu/data";
    public float publishRate = 100.0f; // Hz
    public float noiseLevel = 0.01f; // Noise level for sensor simulation

    [Header("Gravity Settings")]
    public Vector3 gravity = new Vector3(0, -9.81f, 0);

    private ROSConnection ros;
    private float lastPublishTime = 0.0f;
    private Rigidbody rb; // Reference to rigidbody for motion data

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        rb = GetComponentInParent<Rigidbody>(); // IMU is typically attached to a rigidbody
    }

    void Update()
    {
        float currentTime = Time.time;
        if (currentTime - lastPublishTime >= 1.0f / publishRate)
        {
            PublishIMUData();
            lastPublishTime = currentTime;
        }
    }

    void PublishIMUData()
    {
        ImuMsg imuMsg = CreateIMUMessage();
        ros.Publish(rosTopic, imuMsg);
    }

    ImuMsg CreateIMUMessage()
    {
        ImuMsg imuMsg = new ImuMsg();

        imuMsg.header = new HeaderMsg();
        imuMsg.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        imuMsg.header.frame_id = frameId;

        // Orientation (from Unity rotation)
        Quaternion unityRotation = transform.rotation;
        imuMsg.orientation = new QuaternionMsg(
            unityRotation.x,
            unityRotation.y,
            unityRotation.z,
            unityRotation.w
        );

        // Orientation covariance (set to -1 if not available)
        for (int i = 0; i < 9; i++)
        {
            imuMsg.orientation_covariance[i] = -1.0f;
        }

        // Angular velocity (from rigidbody if available)
        if (rb != null)
        {
            Vector3 angularVel = rb.angularVelocity;
            // Add noise to simulate real sensor
            angularVel += AddNoise(angularVel, noiseLevel * 0.1f);

            imuMsg.angular_velocity = new Vector3Msg(
                angularVel.x,
                angularVel.y,
                angularVel.z
            );
        }
        else
        {
            // If no rigidbody, estimate from rotation change
            imuMsg.angular_velocity = new Vector3Msg(0, 0, 0);
        }

        // Angular velocity covariance
        for (int i = 0; i < 9; i++)
        {
            imuMsg.angular_velocity_covariance[i] = -1.0f;
        }

        // Linear acceleration
        Vector3 linearAccel = gravity; // Start with gravity

        if (rb != null)
        {
            // Add acceleration from rigidbody movement
            linearAccel += rb.acceleration;
        }

        // Transform to IMU frame
        linearAccel = transform.InverseTransformDirection(linearAccel);

        // Add noise
        linearAccel += AddNoise(linearAccel, noiseLevel);

        imuMsg.linear_acceleration = new Vector3Msg(
            linearAccel.x,
            linearAccel.y,
            linearAccel.z
        );

        // Linear acceleration covariance
        for (int i = 0; i < 9; i++)
        {
            imuMsg.linear_acceleration_covariance[i] = -1.0f;
        }

        return imuMsg;
    }

    // Add Gaussian noise to simulate real sensor characteristics
    Vector3 AddNoise(Vector3 value, float noiseLevel)
    {
        return new Vector3(
            value.x + RandomGaussian() * noiseLevel,
            value.y + RandomGaussian() * noiseLevel,
            value.z + RandomGaussian() * noiseLevel
        );
    }

    // Generate Gaussian random number using Box-Muller transform
    float RandomGaussian()
    {
        float u1 = Random.value;
        float u2 = Random.value;
        if (u1 < Mathf.Epsilon) u1 = Mathf.Epsilon; // Avoid log(0)
        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
    }
}
```

## Advanced Integration Patterns

### State Publisher

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Nav;
using RosMessageTypes.Tf2;

public class UnityStatePublisher : MonoBehaviour
{
    [Header("Publishing Settings")]
    public float publishRate = 50.0f; // Hz
    public string odomTopic = "/odom";
    public string tfTopic = "/tf";

    private ROSConnection ros;
    private float lastPublishTime = 0.0f;
    private Transform previousTransform;
    private float previousTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        previousTransform = transform;
        previousTime = Time.time;
    }

    void Update()
    {
        float currentTime = Time.time;
        if (currentTime - lastPublishTime >= 1.0f / publishRate)
        {
            PublishState();
            lastPublishTime = currentTime;
        }
    }

    void PublishState()
    {
        // Publish odometry
        PublishOdometry();

        // Publish transform
        PublishTransform();
    }

    void PublishOdometry()
    {
        var odom = new OdometryMsg();

        odom.header = new HeaderMsg();
        odom.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";

        // Position
        odom.pose.pose.position = new Vector3Msg(transform.position.x, transform.position.y, transform.position.z);

        // Orientation
        Quaternion rot = transform.rotation;
        odom.pose.pose.orientation = new QuaternionMsg(rot.x, rot.y, rot.z, rot.w);

        // Calculate velocity from position change
        float deltaTime = Time.time - previousTime;
        if (deltaTime > 0)
        {
            Vector3 velocity = (transform.position - previousTransform.position) / deltaTime;
            odom.twist.twist.linear = new Vector3Msg(velocity.x, velocity.y, velocity.z);

            // Calculate angular velocity from rotation change
            Quaternion rotChange = transform.rotation * Quaternion.Inverse(previousTransform.rotation);
            Vector3 angularVelocity = new Vector3(
                Mathf.Atan2(2 * (rotChange.y * rotChange.z + rotChange.w * rotChange.x),
                           rotChange.w * rotChange.w - rotChange.x * rotChange.x - rotChange.y * rotChange.y + rotChange.z * rotChange.z),
                Mathf.Asin(-2 * (rotChange.x * rotChange.z - rotChange.w * rotChange.y)),
                Mathf.Atan2(2 * (rotChange.x * rotChange.y + rotChange.w * rotChange.z),
                           rotChange.w * rotChange.w + rotChange.x * rotChange.x - rotChange.y * rotChange.y - rotChange.z * rotChange.z)
            ) / deltaTime;

            odom.twist.twist.angular = new Vector3Msg(angularVelocity.x, angularVelocity.y, angularVelocity.z);
        }

        ros.Publish(odomTopic, odom);

        // Update for next calculation
        previousTransform = transform;
        previousTime = Time.time;
    }

    void PublishTransform()
    {
        var tf = new TfMessage();

        var transformStamped = new TransformStampedMsg();
        transformStamped.header = new HeaderMsg();
        transformStamped.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        transformStamped.header.frame_id = "odom";
        transformStamped.child_frame_id = "base_link";

        // Translation
        transformStamped.transform.translation = new Vector3Msg(
            transform.position.x,
            transform.position.y,
            transform.position.z
        );

        // Rotation
        Quaternion rot = transform.rotation;
        transformStamped.transform.rotation = new QuaternionMsg(rot.x, rot.y, rot.z, rot.w);

        tf.transforms = new TransformStampedMsg[1] { transformStamped };

        ros.Publish(tfTopic, tf);
    }
}
```

## Error Handling and Diagnostics

### Connection Monitoring

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class ROSConnectionMonitor : MonoBehaviour
{
    [Header("Connection Settings")]
    public float connectionCheckInterval = 1.0f;
    public int maxConnectionAttempts = 5;

    private ROSConnection ros;
    private float lastCheckTime = 0.0f;
    private int connectionAttempts = 0;
    private bool isConnected = false;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
    }

    void Update()
    {
        float currentTime = Time.time;
        if (currentTime - lastCheckTime >= connectionCheckInterval)
        {
            CheckConnection();
            lastCheckTime = currentTime;
        }
    }

    void CheckConnection()
    {
        if (ros != null)
        {
            // Check if connection is still active
            bool currentConnectionStatus = ros.IsConnected;

            if (currentConnectionStatus != isConnected)
            {
                if (currentConnectionStatus)
                {
                    Debug.Log("ROS connection established");
                    OnConnectionEstablished();
                }
                else
                {
                    Debug.LogWarning("ROS connection lost");
                    OnConnectionLost();
                }
            }

            isConnected = currentConnectionStatus;
        }
    }

    void OnConnectionEstablished()
    {
        connectionAttempts = 0;
        // Re-register publishers/subscribers if needed
    }

    void OnConnectionLost()
    {
        connectionAttempts++;

        if (connectionAttempts < maxConnectionAttempts)
        {
            Debug.Log($"Attempting to reconnect... ({connectionAttempts}/{maxConnectionAttempts})");
            // Try to reconnect
            ros.Initialize();
        }
        else
        {
            Debug.LogError("Max connection attempts reached. Connection failed.");
            OnConnectionFailed();
        }
    }

    void OnConnectionFailed()
    {
        // Handle connection failure - stop simulation, log error, etc.
        Debug.LogError("ROS connection failed permanently");
    }

    // Test connection by publishing a heartbeat message
    public bool TestConnection()
    {
        try
        {
            // Try to publish a simple message
            var timeMsg = new RosMessageTypes.Std.TimeMsg();
            timeMsg.data = new RosMessageTypes.Std.TimeStamp(ROSConnection.GetServerTime());

            // This is just a test - in practice, you'd publish to a test topic
            return ros != null && ros.IsConnected;
        }
        catch
        {
            return false;
        }
    }
}
```

## Best Practices for Unity-ROS Integration

### 1. Performance Optimization
- Use appropriate publish rates for different sensor types
- Implement object pooling for message creation
- Use efficient data structures for sensor data
- Consider running Unity simulation at fixed timestep

### 2. Data Consistency
- Synchronize Unity time with ROS time when possible
- Use appropriate coordinate frame conventions
- Ensure data types match between Unity and ROS
- Implement proper timestamping for sensor data

### 3. Error Handling
- Implement robust connection monitoring
- Handle message serialization errors gracefully
- Provide fallback behaviors when ROS connection is lost
- Log errors for debugging and monitoring

### 4. Architecture
- Separate simulation logic from ROS communication
- Use event-driven patterns for message handling
- Implement modular components for different sensors
- Design for scalability and maintainability

## Learning Objectives

By the end of this lesson, you should be able to:
- Set up and configure ROS-TCP-Connector for Unity integration
- Implement robot control interfaces using ROS messages
- Simulate various sensors (camera, LiDAR, IMU) and publish data to ROS
- Handle connection monitoring and error recovery
- Apply best practices for Unity-ROS integration
- Design modular and scalable integration architectures
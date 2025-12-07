---
title: 'Unity for Realistic Physics Simulation'
description: 'Utilizing Unity''s physics engine for realistic robot simulation, explaining Unity''s physics engine for simulating interactions, forces, materials, rigid bodies, colliders, and joints'
chapter: 6
lesson: 2
module: 2
sidebar_label: 'Unity for Realistic Physics Simulation'
sidebar_position: 2
tags: ['Unity', 'Physics', 'Simulation', 'Rigid Bodies', 'Colliders', 'Joints']
keywords: ['Unity', 'physics engine', 'rigid bodies', 'colliders', 'joints', 'forces', 'materials', 'robotics simulation']
---

# Unity for Realistic Physics Simulation

## Overview

Unity's physics engine, based on NVIDIA's PhysX, provides a comprehensive system for simulating realistic physical interactions in robotic applications. Understanding how to properly configure rigid bodies, colliders, joints, and materials is crucial for creating accurate simulations that closely match real-world robot behavior. This lesson explores Unity's physics capabilities and how to leverage them for realistic robotics simulation.

## Unity Physics Engine Fundamentals

### Physics Engine Architecture

Unity's physics system operates on a fixed timestep to ensure consistent simulation across different hardware:

```csharp
using UnityEngine;

public class PhysicsConfiguration : MonoBehaviour
{
    [Header("Physics Settings")]
    public float fixedTimestep = 0.02f; // 50 Hz
    public float maxAllowedTimestep = 0.1f;
    public int solverIterations = 6;
    public int solverVelocityIterations = 1;

    void Start()
    {
        ConfigurePhysics();
    }

    void ConfigurePhysics()
    {
        // Set the fixed timestep for physics calculations
        Time.fixedDeltaTime = fixedTimestep;

        // Maximum time allowed for physics updates per frame
        Time.maxDeltaTime = maxAllowedTimestep;

        // Configure solver iterations for accuracy
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;

        // Set gravity (default is -9.81 in Y direction)
        Physics.gravity = new Vector3(0, -9.81f, 0);

        // Configure bounce threshold (minimum velocity for bouncing)
        Physics.bounceThreshold = 2.0f;

        // Sleep threshold (minimum energy for objects to sleep)
        Physics.sleepThreshold = 0.005f;
    }
}
```

### Physics Update Cycle

Unity's physics simulation follows this cycle:
1. **Collision Detection**: Detect collisions between colliders
2. **Constraint Solving**: Solve joint constraints and collision responses
3. **Integration**: Update positions and velocities based on forces
4. **Broad Phase**: Determine potential collision pairs
5. **Narrow Phase**: Precise collision detection and response

## Rigid Body Components

### Rigid Body Configuration

The Rigidbody component is essential for physics simulation:

```csharp
using UnityEngine;

public class RobotRigidBodySetup : MonoBehaviour
{
    [Header("Rigid Body Properties")]
    public float mass = 1.0f;
    public Vector3 centerOfMass = Vector3.zero;
    public Vector3 inertiaTensor = Vector3.one;
    public bool useGravity = true;
    public bool isKinematic = false;

    [Header("Drag Properties")]
    public float drag = 0.0f;
    public float angularDrag = 0.0f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        ConfigureRigidBody();
    }

    void ConfigureRigidBody()
    {
        // Set mass
        rb.mass = mass;

        // Set center of mass
        rb.centerOfMass = centerOfMass;

        // Set inertia tensor (how mass is distributed)
        rb.inertiaTensor = inertiaTensor;

        // Configure gravity
        rb.useGravity = useGravity;

        // Kinematic bodies are moved by animation or script, not physics
        rb.isKinematic = isKinematic;

        // Set drag properties
        rb.drag = drag;
        rb.angularDrag = angularDrag;

        // Configure collision detection mode
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
    }

    // Apply forces to the rigid body
    public void ApplyForce(Vector3 force, ForceMode mode = ForceMode.Force)
    {
        rb.AddForce(force, mode);
    }

    // Apply torque to the rigid body
    public void ApplyTorque(Vector3 torque, ForceMode mode = ForceMode.Force)
    {
        rb.AddTorque(torque, mode);
    }

    // Apply force at a specific position
    public void ApplyForceAtPosition(Vector3 force, Vector3 position, ForceMode mode = ForceMode.Force)
    {
        rb.AddForceAtPosition(force, position, mode);
    }
}
```

### Rigid Body Constraints

Rigid bodies can be constrained to limit movement:

```csharp
using UnityEngine;

public class RigidBodyConstraints : MonoBehaviour
{
    [Header("Position Constraints")]
    public bool freezePositionX = false;
    public bool freezePositionY = false;
    public bool freezePositionZ = false;

    [Header("Rotation Constraints")]
    public bool freezeRotationX = false;
    public bool freezeRotationY = false;
    public bool freezeRotationZ = false;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        ApplyConstraints();
    }

    void ApplyConstraints()
    {
        // Set position constraints
        rb.constraints = RigidbodyConstraints.None;

        if (freezePositionX) rb.constraints |= RigidbodyConstraints.FreezePositionX;
        if (freezePositionY) rb.constraints |= RigidbodyConstraints.FreezePositionY;
        if (freezePositionZ) rb.constraints |= RigidbodyConstraints.FreezePositionZ;

        if (freezeRotationX) rb.constraints |= RigidbodyConstraints.FreezeRotationX;
        if (freezeRotationY) rb.constraints |= RigidbodyConstraints.FreezeRotationY;
        if (freezeRotationZ) rb.constraints |= RigidbodyConstraints.FreezeRotationZ;
    }

    // Dynamic constraint changes
    public void SetPositionConstraint(Axis axis, bool freeze)
    {
        switch (axis)
        {
            case Axis.X:
                if (freeze) rb.constraints |= RigidbodyConstraints.FreezePositionX;
                else rb.constraints &= ~RigidbodyConstraints.FreezePositionX;
                break;
            case Axis.Y:
                if (freeze) rb.constraints |= RigidbodyConstraints.FreezePositionY;
                else rb.constraints &= ~RigidbodyConstraints.FreezePositionY;
                break;
            case Axis.Z:
                if (freeze) rb.constraints |= RigidbodyConstraints.FreezePositionZ;
                else rb.constraints &= ~RigidbodyConstraints.FreezePositionZ;
                break;
        }
    }
}

public enum Axis { X, Y, Z }
```

## Collider Components

### Collider Types and Configuration

Unity supports various collider types for different purposes:

```csharp
using UnityEngine;

public class ColliderConfiguration : MonoBehaviour
{
    [Header("Collider Settings")]
    public ColliderType colliderType = ColliderType.Box;
    public bool isTrigger = false;
    public Material physicsMaterial;

    [Header("Box Collider Settings")]
    public Vector3 boxSize = Vector3.one;

    [Header("Sphere Collider Settings")]
    public float sphereRadius = 0.5f;

    [Header("Capsule Collider Settings")]
    public float capsuleRadius = 0.5f;
    public float capsuleHeight = 2.0f;
    public CapsuleDirection capsuleDirection = CapsuleDirection.Y;

    private Collider collider;

    void Start()
    {
        CreateCollider();
    }

    void CreateCollider()
    {
        // Remove existing colliders
        Collider[] existingColliders = GetComponents<Collider>();
        foreach (Collider c in existingColliders)
        {
            DestroyImmediate(c);
        }

        // Create appropriate collider based on type
        switch (colliderType)
        {
            case ColliderType.Box:
                var boxCollider = gameObject.AddComponent<BoxCollider>();
                boxCollider.size = boxSize;
                boxCollider.isTrigger = isTrigger;
                if (physicsMaterial != null) boxCollider.material = physicsMaterial;
                collider = boxCollider;
                break;

            case ColliderType.Sphere:
                var sphereCollider = gameObject.AddComponent<SphereCollider>();
                sphereCollider.radius = sphereRadius;
                sphereCollider.isTrigger = isTrigger;
                if (physicsMaterial != null) sphereCollider.material = physicsMaterial;
                collider = sphereCollider;
                break;

            case ColliderType.Capsule:
                var capsuleCollider = gameObject.AddComponent<CapsuleCollider>();
                capsuleCollider.radius = capsuleRadius;
                capsuleCollider.height = capsuleHeight;
                capsuleCollider.direction = (int)capsuleDirection;
                capsuleCollider.isTrigger = isTrigger;
                if (physicsMaterial != null) capsuleCollider.material = physicsMaterial;
                collider = capsuleCollider;
                break;

            case ColliderType.Mesh:
                var meshCollider = gameObject.AddComponent<MeshCollider>();
                meshCollider.convex = true; // Use convex hull for dynamic objects
                meshCollider.isTrigger = isTrigger;
                if (physicsMaterial != null) meshCollider.material = physicsMaterial;
                collider = meshCollider;
                break;
        }
    }

    // Adjust collider size at runtime
    public void UpdateColliderSize(Vector3 newSize)
    {
        if (collider is BoxCollider box)
        {
            box.size = newSize;
        }
    }

    // Enable/disable collision
    public void SetColliderEnabled(bool enabled)
    {
        if (collider != null)
        {
            collider.enabled = enabled;
        }
    }
}

public enum ColliderType { Box, Sphere, Capsule, Mesh }
```

### Physics Materials

Physics materials define surface properties:

```csharp
using UnityEngine;

[CreateAssetMenu(fileName = "PhysicsMaterial", menuName = "Robotics/Physics Material")]
public class RobotPhysicsMaterial : ScriptableObject
{
    [Header("Friction Properties")]
    [Tooltip("How much the surface resists sliding")]
    [Range(0f, 1f)]
    public float staticFriction = 0.6f;

    [Range(0f, 1f)]
    public float dynamicFriction = 0.5f;

    [Header("Bounce Properties")]
    [Tooltip("How bouncy the surface is (0 = no bounce, 1 = very bouncy)")]
    [Range(0f, 1f)]
    public float bounciness = 0.2f;

    [Header("Friction Combine")]
    public PhysicMaterialCombine frictionCombine = PhysicMaterialCombine.Average;

    [Header("Bounce Combine")]
    public PhysicMaterialCombine bounceCombine = PhysicMaterialCombine.Average;

    // Create Unity PhysicMaterial from these settings
    public PhysicMaterial CreatePhysicMaterial()
    {
        PhysicMaterial material = new PhysicMaterial(name);
        material.staticFriction = staticFriction;
        material.dynamicFriction = dynamicFriction;
        material.bounciness = bounciness;
        material.frictionCombine = frictionCombine;
        material.bounceCombine = bounceCombine;
        return material;
    }
}

// Usage example
public class MaterialApplier : MonoBehaviour
{
    public RobotPhysicsMaterial robotMaterial;
    public RobotPhysicsMaterial groundMaterial;

    void Start()
    {
        ApplyMaterials();
    }

    void ApplyMaterials()
    {
        // Apply material to robot parts
        Collider[] robotColliders = GetComponentsInChildren<Collider>();
        foreach (Collider c in robotColliders)
        {
            if (robotMaterial != null)
            {
                c.material = robotMaterial.CreatePhysicMaterial();
            }
        }

        // Apply different material to ground
        GameObject ground = GameObject.FindGameObjectWithTag("Ground");
        if (ground != null && groundMaterial != null)
        {
            Collider groundCollider = ground.GetComponent<Collider>();
            if (groundCollider != null)
            {
                groundCollider.material = groundMaterial.CreatePhysicMaterial();
            }
        }
    }
}
```

## Joint Components

### Joint Types and Configuration

Unity provides several joint types for connecting rigid bodies:

```csharp
using UnityEngine;

public class RobotJointSetup : MonoBehaviour
{
    [Header("Joint Configuration")]
    public JointType jointType = JointType.Hinge;
    public Transform connectedBody; // The body this joint connects to
    public Vector3 anchor = Vector3.zero; // Anchor point in local space
    public Vector3 axis = Vector3.right; // Axis of rotation for hinge joints

    [Header("Joint Limits")]
    public bool useLimits = true;
    public float lowLimit = -45f;
    public float highLimit = 45f;
    public float bounciness = 0f;

    [Header("Joint Drive")]
    public bool useDrive = true;
    public float drivePosition = 0f;
    public float driveSpring = 10000f;
    public float driveDamper = 100f;
    public float driveMaximumForce = 50f;

    private Joint joint;

    void Start()
    {
        CreateJoint();
    }

    void CreateJoint()
    {
        // Ensure this GameObject has a Rigidbody
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
            rb.isKinematic = true; // Set to kinematic initially
        }

        switch (jointType)
        {
            case JointType.Hinge:
                joint = CreateHingeJoint(rb);
                break;
            case JointType.Fixed:
                joint = CreateFixedJoint(rb);
                break;
            case JointType.Slider:
                joint = CreateSliderJoint(rb);
                break;
            case JointType.Spherical:
                joint = CreateSphericalJoint(rb);
                break;
            case JointType.Configurable:
                joint = CreateConfigurableJoint(rb);
                break;
        }

        if (joint != null)
        {
            ConfigureJointProperties();
        }
    }

    Joint CreateHingeJoint(Rigidbody rb)
    {
        HingeJoint hingeJoint = gameObject.AddComponent<HingeJoint>();
        hingeJoint.connectedBody = connectedBody ? connectedBody.GetComponent<Rigidbody>() : null;
        hingeJoint.anchor = anchor;
        hingeJoint.axis = axis;

        if (useLimits)
        {
            JointLimits limits = hingeJoint.limits;
            limits.min = lowLimit;
            limits.max = highLimit;
            limits.bounciness = bounciness;
            hingeJoint.limits = limits;
        }

        if (useDrive)
        {
            JointDrive drive = hingeJoint.spring;
            drive.position = drivePosition;
            drive.spring = driveSpring;
            drive.damper = driveDamper;
            drive.maximumForce = driveMaximumForce;
            hingeJoint.spring = drive;
        }

        return hingeJoint;
    }

    Joint CreateFixedJoint(Rigidbody rb)
    {
        FixedJoint fixedJoint = gameObject.AddComponent<FixedJoint>();
        fixedJoint.connectedBody = connectedBody ? connectedBody.GetComponent<Rigidbody>() : null;
        return fixedJoint;
    }

    Joint CreateSliderJoint(Rigidbody rb)
    {
        SliderJoint sliderJoint = gameObject.AddComponent<SliderJoint>();
        sliderJoint.connectedBody = connectedBody ? connectedBody.GetComponent<Rigidbody>() : null;
        sliderJoint.anchor = anchor;
        sliderJoint.axis = axis;

        if (useLimits)
        {
            JointLimits limits = sliderJoint.limits;
            limits.min = lowLimit;
            limits.max = highLimit;
            limits.bounciness = bounciness;
            sliderJoint.limits = limits;
        }

        return sliderJoint;
    }

    Joint CreateConfigurableJoint(Rigidbody rb)
    {
        ConfigurableJoint configurableJoint = gameObject.AddComponent<ConfigurableJoint>();
        configurableJoint.connectedBody = connectedBody ? connectedBody.GetComponent<Rigidbody>() : null;
        configurableJoint.anchor = anchor;
        configurableJoint.axis = axis;

        // Configure motion constraints
        configurableJoint.xMotion = ConfigurableJointMotion.Locked;
        configurableJoint.yMotion = ConfigurableJointMotion.Locked;
        configurableJoint.zMotion = ConfigurableJointMotion.Locked;

        configurableJoint.angularXMotion = ConfigurableJointMotion.Limited;
        configurableJoint.angularYMotion = ConfigurableJointMotion.Limited;
        configurableJoint.angularZMotion = ConfigurableJointMotion.Limited;

        if (useLimits)
        {
            SoftJointLimit lowAngularXLimit = configurableJoint.lowAngularXLimit;
            lowAngularXLimit.limit = lowLimit * Mathf.Deg2Rad;
            configurableJoint.lowAngularXLimit = lowAngularXLimit;

            SoftJointLimit highAngularXLimit = configurableJoint.highAngularXLimit;
            highAngularXLimit.limit = highLimit * Mathf.Deg2Rad;
            configurableJoint.highAngularXLimit = highAngularXLimit;
        }

        if (useDrive)
        {
            JointDrive slerpDrive = configurableJoint.slerpDrive;
            slerpDrive.positionSpring = driveSpring;
            slerpDrive.positionDamper = driveDamper;
            slerpDrive.maximumForce = driveMaximumForce;
            configurableJoint.slerpDrive = slerpDrive;
        }

        return configurableJoint;
    }

    void ConfigureJointProperties()
    {
        if (joint != null)
        {
            // Common joint properties
            joint.enableCollision = false; // Don't collide connected bodies
            joint.breakForce = Mathf.Infinity; // Never break under force
            joint.breakTorque = Mathf.Infinity; // Never break under torque
        }
    }

    // Update joint drive during simulation
    public void UpdateJointDrive(float targetAngle)
    {
        if (joint is ConfigurableJoint configurableJoint)
        {
            JointDrive slerpDrive = configurableJoint.slerpDrive;
            slerpDrive.position = targetAngle;
            configurableJoint.slerpDrive = slerpDrive;
        }
        else if (joint is HingeJoint hingeJoint)
        {
            JointDrive spring = hingeJoint.spring;
            spring.position = targetAngle;
            hingeJoint.spring = spring;
        }
    }
}

public enum JointType { Hinge, Fixed, Slider, Spherical, Configurable }
```

### Advanced Joint Control

For more sophisticated joint control, especially for robotic arms:

```csharp
using UnityEngine;

public class RobotArmController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public RobotJoint[] joints;
    public float moveSpeed = 1.0f;

    [System.Serializable]
    public class RobotJoint
    {
        public Transform jointTransform;
        public Joint joint;
        public float currentAngle;
        public float targetAngle;
        public float minAngle = -90f;
        public float maxAngle = 90f;
        public float maxVelocity = 30f; // degrees per second
    }

    void Start()
    {
        InitializeJoints();
    }

    void InitializeJoints()
    {
        foreach (RobotJoint robotJoint in joints)
        {
            if (robotJoint.jointTransform != null)
            {
                robotJoint.currentAngle = robotJoint.jointTransform.localEulerAngles.y;
                robotJoint.targetAngle = robotJoint.currentAngle;
            }
        }
    }

    void Update()
    {
        MoveJointsToTarget();
    }

    void MoveJointsToTarget()
    {
        foreach (RobotJoint robotJoint in joints)
        {
            if (robotJoint.jointTransform != null)
            {
                // Calculate target angle with limits
                float targetAngle = Mathf.Clamp(robotJoint.targetAngle, robotJoint.minAngle, robotJoint.maxAngle);

                // Smooth movement toward target
                float currentAngle = robotJoint.jointTransform.localEulerAngles.y;
                float angleDifference = Mathf.DeltaAngle(currentAngle, targetAngle);

                // Limit velocity
                float maxChange = robotJoint.maxVelocity * Time.deltaTime;
                float clampedDifference = Mathf.Clamp(angleDifference, -maxChange, maxChange);

                float newAngle = currentAngle + clampedDifference;

                // Apply rotation
                robotJoint.jointTransform.localEulerAngles = new Vector3(
                    robotJoint.jointTransform.localEulerAngles.x,
                    newAngle,
                    robotJoint.jointTransform.localEulerAngles.z
                );

                robotJoint.currentAngle = newAngle;
            }
        }
    }

    // Set target angles for all joints
    public void SetJointTargets(float[] targets)
    {
        for (int i = 0; i < Mathf.Min(joints.Length, targets.Length); i++)
        {
            joints[i].targetAngle = targets[i];
        }
    }

    // Get current joint angles
    public float[] GetCurrentJointAngles()
    {
        float[] angles = new float[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            angles[i] = joints[i].currentAngle;
        }
        return angles;
    }

    // Inverse Kinematics example
    public void MoveEndEffectorTo(Vector3 targetPosition)
    {
        // Simple inverse kinematics implementation
        // In a real application, you'd use more sophisticated algorithms
        Transform endEffector = joints[joints.Length - 1].jointTransform;

        // Calculate direction to target
        Vector3 direction = targetPosition - endEffector.position;

        // Move toward target (simplified)
        foreach (RobotJoint joint in joints)
        {
            if (joint.jointTransform != null)
            {
                Vector3 toTarget = targetPosition - joint.jointTransform.position;
                float angle = Vector3.SignedAngle(
                    joint.jointTransform.forward,
                    toTarget,
                    Vector3.up
                );
                joint.targetAngle += angle * 0.01f; // Small adjustment
            }
        }
    }
}
```

## Forces and Interactions

### Applying Forces and Torques

```csharp
using UnityEngine;

public class ForceController : MonoBehaviour
{
    [Header("Force Properties")]
    public float forceMagnitude = 10.0f;
    public ForceMode forceMode = ForceMode.Force;
    public Vector3 forceDirection = Vector3.forward;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        // Apply force based on input
        if (Input.GetKey(KeyCode.Space))
        {
            ApplyForce();
        }
    }

    public void ApplyForce()
    {
        if (rb != null)
        {
            Vector3 force = forceDirection * forceMagnitude;
            rb.AddForce(force, forceMode);
        }
    }

    public void ApplyForceAtPosition(Vector3 position)
    {
        if (rb != null)
        {
            Vector3 force = forceDirection * forceMagnitude;
            rb.AddForceAtPosition(force, position, forceMode);
        }
    }

    // Apply impulse for immediate response
    public void ApplyImpulse()
    {
        if (rb != null)
        {
            Vector3 impulse = forceDirection * forceMagnitude;
            rb.AddForce(impulse, ForceMode.Impulse);
        }
    }

    // Apply torque for rotational motion
    public void ApplyTorque()
    {
        if (rb != null)
        {
            Vector3 torque = Vector3.up * forceMagnitude;
            rb.AddTorque(torque, forceMode);
        }
    }

    // Calculate required force to reach a target
    public Vector3 CalculateRequiredForce(Vector3 targetPosition, float timeToTarget = 1.0f)
    {
        if (rb == null) return Vector3.zero;

        Vector3 displacement = targetPosition - transform.position;
        Vector3 currentVelocity = rb.velocity;

        // Simple physics calculation: F = m * a
        // Where a = (vf - vi) / t, and vf = displacement / t
        Vector3 finalVelocity = displacement / timeToTarget;
        Vector3 acceleration = (finalVelocity - currentVelocity) / timeToTarget;

        return rb.mass * acceleration;
    }
}
```

### Collision Detection and Response

```csharp
using UnityEngine;

public class CollisionHandler : MonoBehaviour
{
    [Header("Collision Settings")]
    public bool logCollisions = true;
    public bool triggerCallbacks = true;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void OnCollisionEnter(Collision collision)
    {
        if (triggerCallbacks)
        {
            HandleCollisionEnter(collision);
        }
    }

    void OnCollisionStay(Collision collision)
    {
        if (triggerCallbacks)
        {
            HandleCollisionStay(collision);
        }
    }

    void OnCollisionExit(Collision collision)
    {
        if (triggerCallbacks)
        {
            HandleCollisionExit(collision);
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (triggerCallbacks)
        {
            HandleTriggerEnter(other);
        }
    }

    void OnTriggerStay(Collider other)
    {
        if (triggerCallbacks)
        {
            HandleTriggerStay(other);
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (triggerCallbacks)
        {
            HandleTriggerExit(other);
        }
    }

    void HandleCollisionEnter(Collision collision)
    {
        if (logCollisions)
        {
            Debug.Log($"Collision with {collision.gameObject.name} at {collision.contacts[0].point}");
        }

        // Calculate collision force
        float collisionForce = collision.relativeVelocity.magnitude * rb.mass;
        Debug.Log($"Collision force: {collisionForce}");

        // Apply response based on collision force
        if (collisionForce > 10.0f) // Threshold for significant impact
        {
            OnSignificantImpact(collision);
        }
    }

    void HandleCollisionStay(Collision collision)
    {
        // Handle continuous contact
        foreach (ContactPoint contact in collision.contacts)
        {
            // Apply friction or other continuous forces
            Vector3 frictionForce = -contact.normal * Vector3.Dot(rb.velocity, contact.normal) * 0.1f;
            rb.AddForceAtPosition(frictionForce, contact.point);
        }
    }

    void HandleCollisionExit(Collision collision)
    {
        if (logCollisions)
        {
            Debug.Log($"Collision ended with {collision.gameObject.name}");
        }
    }

    void HandleTriggerEnter(Collider other)
    {
        if (logCollisions)
        {
            Debug.Log($"Trigger entered with {other.name}");
        }
    }

    void HandleTriggerExit(Collider other)
    {
        if (logCollisions)
        {
            Debug.Log($"Trigger exited with {other.name}");
        }
    }

    void OnSignificantImpact(Collision collision)
    {
        // Handle significant impacts (e.g., damage, alerts)
        Debug.LogWarning($"Significant impact detected with {collision.gameObject.name}");

        // Example: Apply impulse in opposite direction
        Vector3 impactDirection = collision.relativeVelocity.normalized;
        rb.AddForce(-impactDirection * 5.0f, ForceMode.Impulse);
    }

    // Calculate impulse response
    public void ApplyImpulseResponse(Collision collision, float responseMultiplier = 1.0f)
    {
        if (rb != null)
        {
            Vector3 impulse = -collision.relativeVelocity * rb.mass * responseMultiplier;
            rb.AddForce(impulse, ForceMode.Impulse);
        }
    }
}
```

## Physics-Based Robot Control

### PID Controller for Joint Control

```csharp
using UnityEngine;

public class JointPIDController : MonoBehaviour
{
    [Header("PID Parameters")]
    public float kp = 100.0f; // Proportional gain
    public float ki = 10.0f;  // Integral gain
    public float kd = 5.0f;   // Derivative gain

    [Header("Joint Settings")]
    public float targetAngle = 0.0f;
    public float maxTorque = 100.0f;

    private Rigidbody rb;
    private float integralError = 0.0f;
    private float previousError = 0.0f;
    private float previousTime = 0.0f;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        previousTime = Time.time;
    }

    void FixedUpdate()
    {
        if (rb != null)
        {
            float currentTime = Time.time;
            float deltaTime = currentTime - previousTime;
            previousTime = currentTime;

            // Calculate error
            float currentAngle = transform.localEulerAngles.y; // Assuming rotation around Y-axis
            float error = Mathf.DeltaAngle(currentAngle, targetAngle);

            // Update integral
            integralError += error * deltaTime;

            // Calculate derivative
            float derivativeError = (error - previousError) / deltaTime;
            previousError = error;

            // Calculate PID output
            float output = kp * error + ki * integralError + kd * derivativeError;

            // Apply torque (limit to max torque)
            float torque = Mathf.Clamp(output, -maxTorque, maxTorque);
            rb.AddTorque(Vector3.up * torque, ForceMode.Force);
        }
    }

    // Set new target angle
    public void SetTargetAngle(float angle)
    {
        targetAngle = angle;
        integralError = 0.0f; // Reset integral to prevent windup
        previousError = 0.0f;
    }

    // Get current error
    public float GetCurrentError()
    {
        float currentAngle = transform.localEulerAngles.y;
        return Mathf.DeltaAngle(currentAngle, targetAngle);
    }
}
```

## Best Practices for Physics Simulation

### 1. Performance Optimization
- Use appropriate collision detection modes
- Configure fixed timestep for desired accuracy
- Use compound colliders instead of complex mesh colliders for dynamic objects
- Set appropriate solver iteration counts

### 2. Accuracy Considerations
- Use fixed timesteps that match your control loop
- Configure mass properties accurately
- Use appropriate drag and angular drag values
- Consider using continuous collision detection for fast-moving objects

### 3. Stability
- Ensure proper mass ratios between connected bodies
- Use appropriate joint limits and drives
- Avoid extremely light or heavy objects in the same simulation
- Test with various time scales and configurations

## Learning Objectives

By the end of this lesson, you should be able to:
- Configure Unity's physics engine for robotics simulation
- Set up and control rigid bodies with appropriate properties
- Implement various collider types and physics materials
- Create and control joints for articulated robots
- Apply forces and torques for robot control
- Handle collision detection and response
- Implement PID control for joint positioning
- Apply best practices for physics simulation accuracy and performance
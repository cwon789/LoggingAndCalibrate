#!/usr/bin/env python3
"""
Advanced odometry simulation with independent noise models for wheel and LiDAR odometry.

This script generates realistic test data by adding configurable noise to both
wheel odometry (control inputs) and LiDAR odometry (relative motions).
"""

import numpy as np
import matplotlib.pyplot as plt

# Ground-Truth Extrinsic offset of LiDAR from base_frame
# (dx, dy, d_theta) in meters and radians
EXTRINSIC = (0.5603, 0.0, np.radians(0.0))  # LiDAR is 30cm forward, 10cm left, rotated 15 degrees

# Motion Parameters
linear_velocity = 0.5      # m/s
forward_duration = 10.0     # seconds
backward_duration = 10.0    # seconds
angular_velocity = 0.5     # rad/s
figure_eight_scale = 1.0   # parameter controlling the size of the '8'

# Simulation Time Step
dt = 0.01  # seconds

# NOISE CONFIGURATION
# Wheel Odometry Noise: noise on control inputs (v, w)
# This simulates wheel slippage, encoder errors, etc.
WHEEL_ODOM_NOISE_STD = (0.00, 0.00)  # (v_std in m/s, w_std in rad/s)

# LiDAR Odometry Noise: noise on relative pose measurements
# This simulates SLAM algorithm errors
LIDAR_ODOM_NOISE_STD = (0.004, 0.004, 0.0001)  # (x_std in m, y_std in m, theta_std in rad)


def create_transformation_matrix(x, y, theta):
    """Create a 3x3 transformation matrix from pose (x, y, theta)."""
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])


def extract_pose_from_matrix(T):
    """Extract (x, y, theta) from a 3x3 transformation matrix."""
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta


def decompose_transformation(T):
    """Decompose a transformation matrix into (dx, dy, dtheta)."""
    return extract_pose_from_matrix(T)


def generate_control_sequence():
    """Generate the master sequence of control commands for four phases."""
    controls = []  # List of (time, linear_v, angular_w) tuples
    t = 0.0
    
    # Phase 1: Go Straight
    print("Phase 1: Go Straight")
    while t < forward_duration:
        controls.append((t, linear_velocity, 0.0))
        t += dt
    
    # Phase 2: CW 360° Rotation
    print("Phase 2: CW 360° Rotation")
    rotation_start_time = t
    rotation_duration = 2 * np.pi / angular_velocity  # Time for 360 degrees
    while t < rotation_start_time + rotation_duration:
        controls.append((t, 0.0, -angular_velocity))
        t += dt
    
    # Phase 3: Go Backward
    print("Phase 3: Go Backward")
    backward_start_time = t
    while t < backward_start_time + backward_duration:
        controls.append((t, -linear_velocity, 0.0))
        t += dt
    
    # Phase 4: CCW 360° Rotation
    print("Phase 4: CCW 360° Rotation")
    rotation2_start_time = t
    while t < rotation2_start_time + rotation_duration:
        controls.append((t, 0.0, angular_velocity))
        t += dt
    
    return controls


def generate_wheel_odometry_with_noise(controls):
    """
    Generate wheel odometry path by integrating NOISY control commands.
    Noise is added to the control inputs (v, w) to simulate wheel slippage and encoder errors.
    """
    poses = [(0.0, 0.0, 0.0)]  # Initial pose at origin
    timestamps = [0.0]
    
    x, y, theta = 0.0, 0.0, 0.0
    v_std, w_std = WHEEL_ODOM_NOISE_STD
    
    for i in range(1, len(controls)):
        t, v_ideal, w_ideal = controls[i]
        dt_step = controls[i][0] - controls[i-1][0]
        
        # Add Gaussian noise to control inputs
        v_noisy = v_ideal + np.random.normal(0, v_std)
        w_noisy = w_ideal + np.random.normal(0, w_std)
        
        # Update pose using NOISY controls with unicycle model
        if abs(w_noisy) < 1e-6:  # Straight motion
            x += v_noisy * dt_step * np.cos(theta)
            y += v_noisy * dt_step * np.sin(theta)
        else:  # Curved motion
            # Use exact integration for constant v and w
            delta_theta = w_noisy * dt_step
            x += v_noisy/w_noisy * (np.sin(theta + delta_theta) - np.sin(theta))
            y += v_noisy/w_noisy * (-np.cos(theta + delta_theta) + np.cos(theta))
            theta += delta_theta
        
        # Normalize theta to [-pi, pi]
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        poses.append((x, y, theta))
        timestamps.append(t)
    
    return np.array(timestamps), np.array(poses)


def generate_ground_truth_path(controls):
    """Generate the perfect ground truth path (no noise) for the base frame."""
    poses = [(0.0, 0.0, 0.0)]  # Initial pose at origin
    x, y, theta = 0.0, 0.0, 0.0
    
    for i in range(1, len(controls)):
        t, v, w = controls[i]
        dt_step = controls[i][0] - controls[i-1][0]
        
        # Update pose using IDEAL controls (no noise)
        if abs(w) < 1e-6:  # Straight motion
            x += v * dt_step * np.cos(theta)
            y += v * dt_step * np.sin(theta)
        else:  # Curved motion
            delta_theta = w * dt_step
            x += v/w * (np.sin(theta + delta_theta) - np.sin(theta))
            y += v/w * (-np.cos(theta + delta_theta) + np.cos(theta))
            theta += delta_theta
        
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        poses.append((x, y, theta))
    
    return np.array(poses)


def generate_lidar_odometry_with_noise(timestamps, wheel_poses):
    """
    Generate LiDAR odometry path with noise on relative motions.
    
    NEW SIMPLIFIED LOGIC:
    Step A: Apply extrinsic transformation to wheel odometry poses
    Step B: Add jitter noise to the transformed poses
    """
    dx_ext, dy_ext, dtheta_ext = EXTRINSIC
    x_std, y_std, theta_std = LIDAR_ODOM_NOISE_STD
    
    # Transformation matrix for extrinsic offset
    T_base_to_lidar = create_transformation_matrix(dx_ext, dy_ext, dtheta_ext)
    
    lidar_odom_poses = []
    
    for x_wheel, y_wheel, theta_wheel in wheel_poses:
        # Step A: Apply extrinsic transformation
        # Create transformation matrix for wheel pose
        T_world_to_base = create_transformation_matrix(x_wheel, y_wheel, theta_wheel)
        
        # Apply extrinsic to get ideal LiDAR pose
        T_world_to_lidar = T_world_to_base @ T_base_to_lidar
        x_lidar_ideal, y_lidar_ideal, theta_lidar_ideal = extract_pose_from_matrix(T_world_to_lidar)
        
        # Step B: Add jitter noise to the ideal LiDAR pose
        x_lidar = x_lidar_ideal + np.random.normal(0, x_std)
        y_lidar = y_lidar_ideal + np.random.normal(0, y_std)
        theta_lidar = theta_lidar_ideal + np.random.normal(0, theta_std)
        
        # Normalize theta to [-pi, pi]
        theta_lidar = np.arctan2(np.sin(theta_lidar), np.cos(theta_lidar))
        
        lidar_odom_poses.append((x_lidar, y_lidar, theta_lidar))
    
    return np.array(lidar_odom_poses)


def plot_paths(wheel_poses, lidar_poses, ground_truth_poses):
    """Plot all paths for comparison, including ground truth."""
    plt.figure(figsize=(12, 9))
    
    # Extract x, y coordinates
    wheel_x = wheel_poses[:, 0]
    wheel_y = wheel_poses[:, 1]
    lidar_x = lidar_poses[:, 0]
    lidar_y = lidar_poses[:, 1]
    gt_x = ground_truth_poses[:, 0]
    gt_y = ground_truth_poses[:, 1]
    
    # Plot paths
    plt.plot(gt_x, gt_y, 'k--', linewidth=1.5, label='Ground Truth (base_frame)', alpha=0.5)
    plt.plot(wheel_x, wheel_y, 'b-', linewidth=2, label='Wheel Odometry (with noise)', alpha=0.8)
    plt.plot(lidar_x, lidar_y, 'r-', linewidth=2, label='LiDAR Odometry (with noise)', alpha=0.8)
    
    # Mark start and end points
    plt.plot(wheel_x[0], wheel_y[0], 'go', markersize=10, label='Start')
    plt.plot(wheel_x[-1], wheel_y[-1], 'bo', markersize=10, label='Wheel End')
    plt.plot(lidar_x[-1], lidar_y[-1], 'ro', markersize=10, label='LiDAR End')
    
    # Add arrows to show direction at several points
    arrow_indices = np.linspace(0, len(wheel_poses)-1, 10, dtype=int)
    for idx in arrow_indices[:-1]:
        # Wheel odometry arrows
        if idx+1 < len(wheel_poses):
            dx_wheel = wheel_x[idx+1] - wheel_x[idx]
            dy_wheel = wheel_y[idx+1] - wheel_y[idx]
            if np.sqrt(dx_wheel**2 + dy_wheel**2) > 0.01:
                plt.arrow(wheel_x[idx], wheel_y[idx], dx_wheel*10, dy_wheel*10, 
                         head_width=0.05, head_length=0.05, fc='b', ec='b', alpha=0.5)
        
        # LiDAR odometry arrows
        if idx+1 < len(lidar_poses):
            dx_lidar = lidar_x[idx+1] - lidar_x[idx]
            dy_lidar = lidar_y[idx+1] - lidar_y[idx]
            if np.sqrt(dx_lidar**2 + dy_lidar**2) > 0.01:
                plt.arrow(lidar_x[idx], lidar_y[idx], dx_lidar*10, dy_lidar*10, 
                         head_width=0.05, head_length=0.05, fc='r', ec='r', alpha=0.5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Odometry with Realistic Noise\n' + 
              f'Extrinsic: dx={EXTRINSIC[0]}m, dy={EXTRINSIC[1]}m, dθ={np.degrees(EXTRINSIC[2]):.1f}°\n' +
              f'Wheel Noise (v,w): {WHEEL_ODOM_NOISE_STD}, LiDAR Noise (x,y,θ): {LIDAR_ODOM_NOISE_STD}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('odometry_with_noise.png', dpi=150)
    plt.show()


def save_odometry_to_file(filename, timestamps, poses):
    """Save odometry data to text file in format: timestamp x y theta_in_radians."""
    with open(filename, 'w') as f:
        f.write("# timestamp x y theta_in_radians\n")
        for t, (x, y, theta) in zip(timestamps, poses):
            f.write(f"{t:.3f} {x:.6f} {y:.6f} {theta:.6f}\n")


def main():
    """Main simulation function."""
    print("=== Odometry Simulation with Dual Noise Models ===")
    print(f"Extrinsic offset: dx={EXTRINSIC[0]}m, dy={EXTRINSIC[1]}m, dθ={np.degrees(EXTRINSIC[2]):.1f}°")
    print(f"\nNoise Configuration:")
    print(f"  Wheel Odometry Noise (v_std, w_std): {WHEEL_ODOM_NOISE_STD}")
    print(f"  LiDAR Odometry Noise (x_std, y_std, theta_std): {LIDAR_ODOM_NOISE_STD}")
    
    # Generate control sequence
    print("\nGenerating control sequence...")
    controls = generate_control_sequence()
    print(f"Total simulation time: {controls[-1][0]:.1f} seconds")
    print(f"Total control commands: {len(controls)}")
    
    # Generate ground truth path (no noise)
    print("\nGenerating ground truth path...")
    ground_truth_poses = generate_ground_truth_path(controls)
    
    # Generate wheel odometry with noise on control inputs
    print("Generating wheel odometry with control input noise...")
    timestamps, wheel_poses = generate_wheel_odometry_with_noise(controls)
    
    # Generate LiDAR odometry with noise on relative motions
    print("Generating LiDAR odometry with relative motion noise...")
    lidar_poses = generate_lidar_odometry_with_noise(timestamps, wheel_poses)
    
    # Plot comparison
    print("\nCreating visualization...")
    plot_paths(wheel_poses, lidar_poses, ground_truth_poses)
    
    # Save to files
    print("\nSaving odometry data to files...")
    save_odometry_to_file('wheel_odom.txt', timestamps, wheel_poses)
    save_odometry_to_file('lidar_odom.txt', timestamps, lidar_poses)
    
    print("\nSimulation complete!")
    print("Files saved: wheel_odom.txt, lidar_odom.txt")
    print("Plot saved: odometry_with_noise.png")
    
    # Print statistics
    print("\n=== Noise Statistics ===")
    
    # Calculate actual noise in wheel odometry
    wheel_errors = []
    for i in range(len(wheel_poses)):
        if i < len(ground_truth_poses):
            error_x = wheel_poses[i][0] - ground_truth_poses[i][0]
            error_y = wheel_poses[i][1] - ground_truth_poses[i][1]
            error_dist = np.sqrt(error_x**2 + error_y**2)
            wheel_errors.append(error_dist)
    
    print(f"\nWheel Odometry (due to control noise):")
    print(f"  Mean position error: {np.mean(wheel_errors):.4f} m")
    print(f"  Max position error: {np.max(wheel_errors):.4f} m")
    print(f"  Final position error: {wheel_errors[-1]:.4f} m")
    
    # For LiDAR, we can't directly compare to ground truth since it has extrinsic offset
    # But we can report the final positions
    wheel_end = wheel_poses[-1]
    lidar_end = lidar_poses[-1]
    print(f"\nFinal Positions:")
    print(f"  Wheel: x={wheel_end[0]:.3f}m, y={wheel_end[1]:.3f}m, θ={np.degrees(wheel_end[2]):.1f}°")
    print(f"  LiDAR: x={lidar_end[0]:.3f}m, y={lidar_end[1]:.3f}m, θ={np.degrees(lidar_end[2]):.1f}°")
    
    # Calculate distance traveled
    wheel_distance = 0
    lidar_distance = 0
    for i in range(1, len(wheel_poses)):
        wheel_distance += np.sqrt((wheel_poses[i][0] - wheel_poses[i-1][0])**2 + 
                                 (wheel_poses[i][1] - wheel_poses[i-1][1])**2)
    for i in range(1, len(lidar_poses)):
        lidar_distance += np.sqrt((lidar_poses[i][0] - lidar_poses[i-1][0])**2 + 
                                 (lidar_poses[i][1] - lidar_poses[i-1][1])**2)
    
    print(f"\nTotal Distance Traveled:")
    print(f"  Wheel odometry: {wheel_distance:.3f} m")
    print(f"  LiDAR odometry: {lidar_distance:.3f} m")


if __name__ == "__main__":
    # Set random seed for reproducibility (comment out for different noise each run)
    # np.random.seed(42)
    main()
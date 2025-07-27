#!/usr/bin/env python3
"""
High-Precision 3-DOF Extrinsic Calibration using Non-Linear Least Squares

This script automatically finds the optimal 3-DOF extrinsic calibration (dx, dy, d_theta)
between wheel odometry and LiDAR odometry by optimizing all three parameters simultaneously.
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import sys


def load_odometry_file(filename):
    """Load odometry data from file."""
    try:
        data = np.loadtxt(filename, skiprows=1)  # Skip header
        print(f"Loaded {len(data)} poses from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        print("Please ensure wheel_odom.txt and lidar_odom.txt are in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)


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


def calculate_relative_transformations(odometry_data):
    """
    Calculate relative transformation matrices between consecutive poses.
    Returns a list of 3x3 transformation matrices.
    """
    relative_transforms = []
    
    for i in range(1, len(odometry_data)):
        # Previous pose
        x_prev, y_prev, theta_prev = odometry_data[i-1, 1], odometry_data[i-1, 2], odometry_data[i-1, 3]
        T_prev = create_transformation_matrix(x_prev, y_prev, theta_prev)
        
        # Current pose
        x_curr, y_curr, theta_curr = odometry_data[i, 1], odometry_data[i, 2], odometry_data[i, 3]
        T_curr = create_transformation_matrix(x_curr, y_curr, theta_curr)
        
        # Relative transformation: T_relative = inv(T_prev) * T_curr
        T_relative = np.linalg.inv(T_prev) @ T_curr
        relative_transforms.append(T_relative)
    
    return relative_transforms


def residual_function(x_guess, wheel_relative_transforms, lidar_relative_transforms):
    """
    Residual function for the least squares optimizer.
    
    This function implements the kinematic constraint:
    Error_matrix = inv(T_wheel) * X_guess * T_lidar * inv(X_guess)
    
    When the calibration is perfect, Error_matrix should be the identity matrix.
    
    Args:
        x_guess: Current guess for extrinsic parameters [dx, dy, dtheta]
        wheel_relative_transforms: List of relative transformations from wheel odometry
        lidar_relative_transforms: List of relative transformations from LiDAR odometry
    
    Returns:
        Flattened array of all residual errors (error_x, error_y, error_theta)
    """
    # Create transformation matrix from current extrinsic guess
    dx, dy, dtheta = x_guess
    X_guess = create_transformation_matrix(dx, dy, dtheta)
    X_guess_inv = np.linalg.inv(X_guess)
    
    # List to store all error vectors
    all_errors = []
    
    # Iterate through all relative motion pairs
    for T_wheel, T_lidar in zip(wheel_relative_transforms, lidar_relative_transforms):
        # Calculate error matrix using the kinematic constraint
        # Error_matrix = inv(T_wheel) * X_guess * T_lidar * inv(X_guess)
        # When perfect, this should equal the identity matrix
        Error_matrix = np.linalg.inv(T_wheel) @ X_guess @ T_lidar @ X_guess_inv
        
        # Extract error from identity matrix
        # For a perfect calibration, Error_matrix = I (identity)
        # So we compute: Error = Error_matrix - I
        Identity = np.eye(3)
        Error_diff = Error_matrix - Identity
        
        # Extract error components
        error_x = Error_diff[0, 2]  # x-translation error
        error_y = Error_diff[1, 2]  # y-translation error
        
        # For rotation error, we use the deviation from identity in the rotation part
        # When Error_matrix = I, we have Error_matrix[1,0] = 0 and Error_matrix[0,0] = 1
        # So error_theta = atan2(Error_matrix[1,0], Error_matrix[0,0]) - atan2(0, 1) = atan2(Error_matrix[1,0], Error_matrix[0,0])
        error_theta = np.arctan2(Error_matrix[1, 0], Error_matrix[0, 0])
        
        # Weight the rotation error to make it comparable to translation errors
        # This weight can be tuned based on the relative importance of rotation vs translation
        rotation_weight = 1.0  # 1 radian ≈ 1 meter for unit weighting
        error_theta_weighted = error_theta * rotation_weight
        
        # Append error vector
        all_errors.extend([error_x, error_y, error_theta_weighted])
    
    # Return flattened array of all errors
    return np.array(all_errors)


def apply_extrinsic_to_path(lidar_data, extrinsic):
    """
    Apply the extrinsic transformation to recalculate the LiDAR path.
    
    Args:
        lidar_data: Original LiDAR odometry data
        extrinsic: Extrinsic parameters [dx, dy, dtheta]
    
    Returns:
        Corrected path as numpy array of (x, y, theta) tuples
    """
    dx, dy, dtheta = extrinsic
    X = create_transformation_matrix(dx, dy, dtheta)
    X_inv = np.linalg.inv(X)
    
    # Get original LiDAR poses
    lidar_poses_original = []
    for i in range(len(lidar_data)):
        x, y, theta = lidar_data[i, 1], lidar_data[i, 2], lidar_data[i, 3]
        lidar_poses_original.append((x, y, theta))
    
    # Start with identity transformation for corrected path
    corrected_poses = [(0.0, 0.0, 0.0)]
    T_corrected = np.eye(3)
    
    # For each relative motion in the original LiDAR data
    for i in range(1, len(lidar_poses_original)):
        # Get transformation matrices for consecutive poses
        T_lidar_prev = create_transformation_matrix(*lidar_poses_original[i-1])
        T_lidar_curr = create_transformation_matrix(*lidar_poses_original[i])
        
        # Calculate relative transformation in LiDAR frame
        T_lidar_relative = np.linalg.inv(T_lidar_prev) @ T_lidar_curr
        
        # Apply the kinematic correction: T_wheel = X * T_lidar * X^(-1)
        T_wheel_relative = X @ T_lidar_relative @ X_inv
        
        # Chain the corrected relative transformation
        T_corrected = T_corrected @ T_wheel_relative
        corrected_pose = extract_pose_from_matrix(T_corrected)
        corrected_poses.append(corrected_pose)
    
    return np.array(corrected_poses)


def main():
    """Main calibration routine."""
    print("=== 3-DOF Extrinsic Calibration Solver ===")
    print("Optimizing dx, dy, and d_theta simultaneously\n")
    
    # Load odometry data
    print("Loading odometry data...")
    wheel_data = load_odometry_file('wheel_odom.txt')
    lidar_data = load_odometry_file('lidar_odom.txt')
    
    # Calculate relative transformations
    print("\nCalculating relative transformations...")
    wheel_relative_transforms = calculate_relative_transformations(wheel_data)
    lidar_relative_transforms = calculate_relative_transformations(lidar_data)
    print(f"Number of relative motion pairs: {len(wheel_relative_transforms)}")
    
    # Initial guess for extrinsic parameters [dx, dy, dtheta]
    initial_guess = [0.0, 0.0, 0.0]
    print(f"\nInitial guess: dx={initial_guess[0]:.3f}m, dy={initial_guess[1]:.3f}m, "
          f"dtheta={np.degrees(initial_guess[2]):.1f}°")
    
    # Run the optimizer
    print("\nRunning non-linear least squares optimization...")
    print("Optimizing all three parameters (dx, dy, d_theta) simultaneously...")
    
    # Configure optimizer
    result = least_squares(
        residual_function,
        initial_guess,
        args=(wheel_relative_transforms, lidar_relative_transforms),
        method='lm',  # Levenberg-Marquardt algorithm
        ftol=1e-10,    # Function tolerance
        xtol=1e-10,    # Parameter tolerance
        gtol=1e-10,    # Gradient tolerance
        max_nfev=10000,  # Maximum function evaluations
        verbose=2     # Detailed output
    )

    # result = least_squares(
    #     residual_function,
    #     initial_guess,
    #     args=(wheel_relative_transforms, lidar_relative_transforms),
    #     method='trf',  # <--- 'lm'을 'trf'로 변경
    #     loss='huber',  # 'trf' 메서드와 함께 손실 함수 사용
    #     f_scale=0.1,   # 손실 함수 스케일 (튜닝 필요)
    #     ftol=1e-10,
    #     xtol=1e-10,
    #     gtol=1e-10,
    #     max_nfev=10000,
    #     verbose=2
    # )
    
    # Extract optimized parameters
    optimized_extrinsic = result.x
    dx_opt, dy_opt, dtheta_opt = optimized_extrinsic
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Optimized 3-DOF extrinsic parameters:")
    print(f"  dx (X offset):        {dx_opt:.6f} meters")
    print(f"  dy (Y offset):        {dy_opt:.6f} meters")
    print(f"  d_theta (Rotation):   {np.degrees(dtheta_opt):.4f} degrees")
    print(f"  d_theta (Rotation):   {dtheta_opt:.8f} radians")
    print(f"\nOptimization details:")
    print(f"  Success: {result.success}")
    print(f"  Status: {result.message}")
    print(f"  Final cost (sum of squared residuals): {result.cost:.8f}")
    print(f"  Number of function evaluations: {result.nfev}")
    print(f"  Number of residuals: {len(result.fun)}")
    print(f"  RMS error: {np.sqrt(result.cost / (len(result.fun) / 3)):.6f}")
    print("="*60)
    
    # Generate verification plot
    print("\nGenerating verification plot...")
    
    # Apply optimized extrinsic to LiDAR path
    corrected_lidar_path = apply_extrinsic_to_path(lidar_data, optimized_extrinsic)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Plot original wheel odometry (reference)
    plt.plot(wheel_data[:, 1], wheel_data[:, 2], 'b-', linewidth=2.5, 
             label='Wheel Odometry (Reference)', alpha=0.8)
    
    # Plot corrected LiDAR odometry
    plt.plot(corrected_lidar_path[:, 0], corrected_lidar_path[:, 1], 'g--', linewidth=2.5,
             label='LiDAR Odometry (Calibrated)', alpha=0.8)
    
    # Mark start and end points
    plt.plot(wheel_data[0, 1], wheel_data[0, 2], 'ko', markersize=12, 
             label='Start', zorder=5)
    plt.plot(wheel_data[-1, 1], wheel_data[-1, 2], 'bs', markersize=10, 
             label='Wheel End', zorder=5)
    plt.plot(corrected_lidar_path[-1, 0], corrected_lidar_path[-1, 1], 'g^', markersize=10,
             label='LiDAR End (Calibrated)', zorder=5)
    
    # Calculate final position error
    wheel_end = (wheel_data[-1, 1], wheel_data[-1, 2], wheel_data[-1, 3])
    lidar_end = (corrected_lidar_path[-1, 0], corrected_lidar_path[-1, 1], corrected_lidar_path[-1, 2])
    position_error = np.sqrt((wheel_end[0] - lidar_end[0])**2 + 
                            (wheel_end[1] - lidar_end[1])**2)
    angle_error = np.abs(wheel_end[2] - lidar_end[2])
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'3-DOF Calibration Verification\n' +
              f'Optimized Extrinsic: dx={dx_opt:.4f}m, dy={dy_opt:.4f}m, dθ={np.degrees(dtheta_opt):.2f}°\n' +
              f'Final Position Error: {position_error:.4f}m, Angle Error: {np.degrees(angle_error):.2f}°',
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig('calibration_3dof_result.png', dpi=200)
    print("Verification plot saved as 'calibration_3dof_result.png'")
    plt.show()
    
    # Print path alignment statistics
    print("\nPath Alignment Statistics:")
    total_error = 0.0
    for i in range(len(wheel_data)):
        if i < len(corrected_lidar_path):
            wheel_pos = (wheel_data[i, 1], wheel_data[i, 2])
            lidar_pos = (corrected_lidar_path[i, 0], corrected_lidar_path[i, 1])
            error = np.sqrt((wheel_pos[0] - lidar_pos[0])**2 + 
                           (wheel_pos[1] - lidar_pos[1])**2)
            total_error += error
    
    avg_error = total_error / min(len(wheel_data), len(corrected_lidar_path))
    print(f"  Average position error along path: {avg_error:.4f}m")
    print(f"  Final position error: {position_error:.4f}m")
    print(f"  Final angle error: {np.degrees(angle_error):.2f}°")
    
    print("\nCalibration complete!")


if __name__ == "__main__":
    main()
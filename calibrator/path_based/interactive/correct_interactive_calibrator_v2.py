#!/usr/bin/env python3
"""
Enhanced Interactive Extrinsic Calibration Tool

Features:
- Initial extrinsic values can be set
- 4 decimal places precision display
- Quick step size buttons (0.1, 0.01, 0.001)
- Manual step size input
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import sys


class EnhancedInteractiveCalibrator:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Interactive Calibrator - Transform: (0.0000, 0.0000, 0.0000°)")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load original odometry data and keep it unchanged
        self.wheel_data = self.load_odometry_file('wheel_odom.txt')
        self.lidar_data_original = self.load_odometry_file('lidar_odom.txt')
        
        # Current extrinsic guess
        self.X_guess = {
            'dx': 0.0,
            'dy': 0.0,
            'dtheta': 0.0  # in radians
        }
        
        # Step sizes
        self.move_step = 0.01  # meters
        self.angle_step = 1.0  # degrees
        
        # Create GUI components
        self.create_widgets()
        
        # Initial calculation and plot
        self.recalculate_corrected_path()
        self.update_plot()
        
        # Bind keyboard events to the canvas
        self.canvas.get_tk_widget().bind("<Key>", self.on_key_press)
        self.canvas.get_tk_widget().focus_set()
        
    def load_odometry_file(self, filename):
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
    
    def create_transformation_matrix(self, x, y, theta):
        """Create a 3x3 transformation matrix from pose (x, y, theta)."""
        return np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])
    
    def extract_pose_from_matrix(self, T):
        """Extract (x, y, theta) from a 3x3 transformation matrix."""
        x = T[0, 2]
        y = T[1, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return x, y, theta
    
    def recalculate_corrected_path(self):
        """
        Recalculate the corrected LiDAR path using the current extrinsic guess.
        This implements the correct kinematic transformation.
        """
        # Create transformation matrix for the extrinsic guess
        T_extrinsic = self.create_transformation_matrix(
            self.X_guess['dx'], 
            self.X_guess['dy'], 
            self.X_guess['dtheta']
        )
        T_extrinsic_inv = np.linalg.inv(T_extrinsic)
        
        # Calculate relative transformations from the original LiDAR data
        lidar_poses_original = []
        for i in range(len(self.lidar_data_original)):
            x, y, theta = self.lidar_data_original[i, 1], self.lidar_data_original[i, 2], self.lidar_data_original[i, 3]
            lidar_poses_original.append((x, y, theta))
        
        # Start with identity transformation for corrected path
        corrected_poses = [(0.0, 0.0, 0.0)]
        T_corrected = np.eye(3)
        
        # For each relative motion in the original LiDAR data
        for i in range(1, len(lidar_poses_original)):
            # Get transformation matrices for consecutive poses
            T_lidar_prev = self.create_transformation_matrix(*lidar_poses_original[i-1])
            T_lidar_curr = self.create_transformation_matrix(*lidar_poses_original[i])
            
            # Calculate relative transformation in LiDAR frame
            T_lidar_relative = np.linalg.inv(T_lidar_prev) @ T_lidar_curr
            
            # Apply the kinematic correction: T_wheel = X * T_lidar * X^(-1)
            T_wheel_relative = T_extrinsic @ T_lidar_relative @ T_extrinsic_inv
            
            # Chain the corrected relative transformation
            T_corrected = T_corrected @ T_wheel_relative
            corrected_pose = self.extract_pose_from_matrix(T_corrected)
            corrected_poses.append(corrected_pose)
        
        # Store the corrected path
        self.corrected_path = np.array(corrected_poses)
    
    def create_widgets(self):
        """Create the GUI components."""
        # Main frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel frame
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Initial Extrinsic Values Section
        init_frame = ttk.LabelFrame(control_frame, text="Initial Extrinsic Values", padding="5")
        init_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        # dx input
        ttk.Label(init_frame, text="dx (m):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dx_init_var = tk.StringVar(value="0.0000")
        self.dx_init_entry = ttk.Entry(init_frame, textvariable=self.dx_init_var, width=12)
        self.dx_init_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # dy input
        ttk.Label(init_frame, text="dy (m):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.dy_init_var = tk.StringVar(value="0.0000")
        self.dy_init_entry = ttk.Entry(init_frame, textvariable=self.dy_init_var, width=12)
        self.dy_init_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # dtheta input
        ttk.Label(init_frame, text="dθ (deg):").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.dtheta_init_var = tk.StringVar(value="0.0000")
        self.dtheta_init_entry = ttk.Entry(init_frame, textvariable=self.dtheta_init_var, width=12)
        self.dtheta_init_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Apply initial values button
        self.apply_init_button = ttk.Button(init_frame, text="Apply Initial Values", 
                                          command=self.apply_initial_values)
        self.apply_init_button.grid(row=0, column=2, rowspan=3, padx=10, pady=5)
        
        # Step Size Control Section
        step_frame = ttk.LabelFrame(control_frame, text="Step Size Control", padding="5")
        step_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        # Move step control with quick buttons
        ttk.Label(step_frame, text="Move Step (m):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.move_step_var = tk.StringVar(value=str(self.move_step))
        self.move_step_entry = ttk.Entry(step_frame, textvariable=self.move_step_var, width=10)
        self.move_step_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Quick move step buttons
        move_button_frame = ttk.Frame(step_frame)
        move_button_frame.grid(row=0, column=2, padx=5)
        ttk.Button(move_button_frame, text="0.1", width=5,
                  command=lambda: self.set_move_step(0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(move_button_frame, text="0.01", width=5,
                  command=lambda: self.set_move_step(0.01)).pack(side=tk.LEFT, padx=2)
        ttk.Button(move_button_frame, text="0.001", width=5,
                  command=lambda: self.set_move_step(0.001)).pack(side=tk.LEFT, padx=2)
        
        # Angle step control with quick buttons
        ttk.Label(step_frame, text="Angle Step (deg):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.angle_step_var = tk.StringVar(value=str(self.angle_step))
        self.angle_step_entry = ttk.Entry(step_frame, textvariable=self.angle_step_var, width=10)
        self.angle_step_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Quick angle step buttons
        angle_button_frame = ttk.Frame(step_frame)
        angle_button_frame.grid(row=1, column=2, padx=5)
        ttk.Button(angle_button_frame, text="5.0", width=5,
                  command=lambda: self.set_angle_step(5.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(angle_button_frame, text="1.0", width=5,
                  command=lambda: self.set_angle_step(1.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(angle_button_frame, text="0.1", width=5,
                  command=lambda: self.set_angle_step(0.1)).pack(side=tk.LEFT, padx=2)
        
        # Apply button for manual step sizes
        self.apply_button = ttk.Button(step_frame, text="Apply Steps", command=self.apply_step_sizes)
        self.apply_button.grid(row=0, column=3, rowspan=2, padx=10, pady=5)
        
        # Instructions
        instructions = """
Keyboard Controls (when plot has focus):
• Arrow Keys: Translate extrinsic guess
• '1' Key: Rotate clockwise
• '2' Key: Rotate counter-clockwise

Mouse Controls:
• Use toolbar to pan/zoom
• Scroll wheel to zoom

Note: The path shape changes as you
adjust the extrinsic parameters!
        """
        instruction_label = ttk.Label(control_frame, text=instructions, justify=tk.LEFT)
        instruction_label.grid(row=2, column=0, columnspan=4, pady=10)
        
        # Current transformation display
        self.transform_label = ttk.Label(control_frame, 
                                       text="Current Transform: (0.0000, 0.0000, 0.0000°)", 
                                       font=('TkDefaultFont', 10, 'bold'))
        self.transform_label.grid(row=3, column=0, columnspan=4, pady=5)
        
        # Matplotlib figure and canvas
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure grid weights for resizing
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def apply_initial_values(self):
        """Apply the initial extrinsic values from the input fields."""
        try:
            # Get values from input fields
            dx = float(self.dx_init_var.get())
            dy = float(self.dy_init_var.get())
            dtheta_deg = float(self.dtheta_init_var.get())
            dtheta_rad = np.radians(dtheta_deg)
            
            # Update the extrinsic guess
            self.X_guess['dx'] = dx
            self.X_guess['dy'] = dy
            self.X_guess['dtheta'] = dtheta_rad
            
            print(f"Applied initial extrinsic values: dx={dx:.4f}m, dy={dy:.4f}m, dθ={dtheta_deg:.4f}°")
            
            # Recalculate and update plot
            self.recalculate_corrected_path()
            self.update_plot()
            
        except ValueError as e:
            print(f"Error: Invalid initial values. Please enter numeric values.")
            print(f"Details: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {str(e)}")
        finally:
            # Return focus to canvas
            self.canvas.get_tk_widget().focus_set()
    
    def set_move_step(self, value):
        """Set move step to a specific value."""
        self.move_step = value
        self.move_step_var.set(str(value))
        print(f"Move step set to: {value}m")
        self.canvas.get_tk_widget().focus_set()
    
    def set_angle_step(self, value):
        """Set angle step to a specific value."""
        self.angle_step = value
        self.angle_step_var.set(str(value))
        print(f"Angle step set to: {value}°")
        self.canvas.get_tk_widget().focus_set()
    
    def apply_step_sizes(self):
        """Apply new step sizes from the text entries."""
        try:
            # Get the values from the input boxes
            new_move_step = float(self.move_step_var.get())
            new_angle_step = float(self.angle_step_var.get())
            
            # Check for zero values and warn user
            if new_move_step == 0.0:
                print("Warning: Move step is set to 0. Arrow keys will not move the plot.")
            if new_angle_step == 0.0:
                print("Warning: Angle step is set to 0. Rotation keys will not rotate the plot.")
            
            # Apply the new values
            self.move_step = new_move_step
            self.angle_step = new_angle_step
            print(f"Updated step sizes: move={self.move_step}m, angle={self.angle_step}°")
            
        except ValueError as e:
            # Handle empty or non-numeric input
            print(f"Error: Invalid step size entered. Please enter numeric values.")
            print(f"Details: {str(e)}")
            # Reset display to current valid values
            self.move_step_var.set(str(self.move_step))
            self.angle_step_var.set(str(self.angle_step))
            
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Unexpected error in apply_step_sizes: {type(e).__name__}: {str(e)}")
            # Reset display to current valid values
            self.move_step_var.set(str(self.move_step))
            self.angle_step_var.set(str(self.angle_step))
        
        finally:
            # CRITICAL: Return focus to the canvas so keyboard controls continue working
            self.canvas.get_tk_widget().focus_set()
    
    def update_plot(self):
        """Update the matplotlib plot with current data."""
        self.ax.clear()
        
        # Plot wheel odometry (static reference)
        self.ax.plot(self.wheel_data[:, 1], self.wheel_data[:, 2], 
                    'b-', linewidth=2, label='Wheel Odometry (Reference)', alpha=0.8)
        
        # Plot corrected LiDAR odometry
        self.ax.plot(self.corrected_path[:, 0], self.corrected_path[:, 1], 
                    'r-', linewidth=2, label='LiDAR Odometry (Corrected)', alpha=0.8)
        
        # Mark start and end points
        self.ax.plot(self.wheel_data[0, 1], self.wheel_data[0, 2], 
                    'go', markersize=10, label='Start', zorder=5)
        self.ax.plot(self.wheel_data[-1, 1], self.wheel_data[-1, 2], 
                    'bs', markersize=8, label='Wheel End', zorder=5)
        self.ax.plot(self.corrected_path[-1, 0], self.corrected_path[-1, 1], 
                    'r^', markersize=8, label='LiDAR End', zorder=5)
        
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Enhanced Interactive Extrinsic Calibration\nAlign red path to blue path using keyboard controls')
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')
        self.ax.legend(loc='best')
        
        # Redraw canvas
        self.canvas.draw()
        
        # Update window title and label with current transformation (4 decimal places)
        transform_text = f"({self.X_guess['dx']:.4f}, {self.X_guess['dy']:.4f}, {np.degrees(self.X_guess['dtheta']):.4f}°)"
        self.master.title(f"Enhanced Interactive Calibrator - Transform: {transform_text}")
        self.transform_label.config(text=f"Current Transform: {transform_text}")
        
        # Update the input fields to show current values
        self.dx_init_var.set(f"{self.X_guess['dx']:.4f}")
        self.dy_init_var.set(f"{self.X_guess['dy']:.4f}")
        self.dtheta_init_var.set(f"{np.degrees(self.X_guess['dtheta']):.4f}")
    
    def on_key_press(self, event):
        """Handle keyboard input for transformations."""
        # Get current step sizes in appropriate units
        move_delta = self.move_step
        angle_delta = np.radians(self.angle_step)
        
        # Track if we need to update
        update_needed = False
        
        # Handle different keys
        if event.keysym == 'Up':
            self.X_guess['dy'] += move_delta
            update_needed = True
        elif event.keysym == 'Down':
            self.X_guess['dy'] -= move_delta
            update_needed = True
        elif event.keysym == 'Left':
            self.X_guess['dx'] -= move_delta
            update_needed = True
        elif event.keysym == 'Right':
            self.X_guess['dx'] += move_delta
            update_needed = True
        elif event.char == '1':  # Rotate CW
            self.X_guess['dtheta'] -= angle_delta
            update_needed = True
        elif event.char == '2':  # Rotate CCW
            self.X_guess['dtheta'] += angle_delta
            update_needed = True
        
        # If transformation changed, recalculate and update
        if update_needed:
            self.recalculate_corrected_path()
            self.update_plot()
        
        # Keep focus on canvas for continued keyboard input
        self.canvas.get_tk_widget().focus_set()
    
    def on_closing(self):
        """Handle window closing event."""
        print("\n" + "="*50)
        print("FINAL EXTRINSIC CALIBRATION RESULT:")
        print(f"  X offset: {self.X_guess['dx']:.4f} meters")
        print(f"  Y offset: {self.X_guess['dy']:.4f} meters")
        print(f"  Angular offset: {np.degrees(self.X_guess['dtheta']):.4f} degrees")
        print(f"  Angular offset: {self.X_guess['dtheta']:.8f} radians")
        print("="*50 + "\n")
        
        # Properly close matplotlib figure
        plt.close(self.fig)
        
        # Destroy the Tkinter window
        self.master.quit()
        self.master.destroy()


def main():
    """Main function to run the interactive calibrator."""
    print("Starting Enhanced Interactive Extrinsic Calibrator...")
    print("Loading odometry files...")
    
    root = tk.Tk()
    app = EnhancedInteractiveCalibrator(root)
    
    print("\nCalibrator ready!")
    print("Features:")
    print("- Set initial extrinsic values")
    print("- Quick step size buttons (0.1, 0.01, 0.001)")
    print("- 4 decimal places precision display")
    print("\nClick on the plot to give it focus, then use keyboard controls.")
    print("Notice how the path SHAPE changes as you adjust the extrinsic parameters!")
    print("Close the window when done to see the final calibration result.\n")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
        root.quit()
    
    # Ensure clean exit
    try:
        root.destroy()
    except:
        pass


if __name__ == "__main__":
    main()
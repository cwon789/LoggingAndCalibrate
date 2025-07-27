#!/usr/bin/env python3
"""
Advanced Marker Calibrator for 2D LiDAR Extrinsic Calibration
This tool allows manual calibration of a 2D LiDAR's pose relative to the robot's base_link
by aligning multiple virtual L-shaped markers with real laser scan data.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QGroupBox, QGridLayout, QTextEdit, QListWidget,
                           QListWidgetItem, QSplitter, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import threading
import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Marker:
    """Data class for L-shaped marker properties"""
    name: str
    x: float = 1.0
    y: float = 0.0
    theta: float = 0.0  # in radians
    segment1_length: float = 0.5
    segment2_length: float = 0.5
    

class LaserScanSubscriber(Node):
    """ROS2 node for subscribing to LaserScan messages with distance filtering"""
    
    def __init__(self, topic_name='/scan'):
        super().__init__('laser_scan_subscriber')
        self.scan_data = None
        self.raw_scan_data = None
        self.subscription = None
        self.topic_name = topic_name
        self.min_distance = 0.1
        self.max_distance = 10.0
        
    def start_subscription(self, topic_name):
        """Start subscribing to the specified topic"""
        if self.subscription:
            self.destroy_subscription(self.subscription)
        
        self.topic_name = topic_name
        self.subscription = self.create_subscription(
            LaserScan,
            self.topic_name,
            self.scan_callback,
            10
        )
        self.get_logger().info(f'Subscribed to {self.topic_name}')
        
    def scan_callback(self, msg):
        """Process incoming LaserScan messages"""
        ranges = np.array(msg.ranges)
        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)
        
        # Filter out invalid readings
        valid_indices = np.where((ranges > msg.range_min) & (ranges < msg.range_max))[0]
        
        if len(valid_indices) > 0:
            valid_ranges = ranges[valid_indices]
            valid_angles = angles[valid_indices]
            
            # Convert polar to Cartesian coordinates
            x = valid_ranges * np.cos(valid_angles)
            y = valid_ranges * np.sin(valid_angles)
            
            self.raw_scan_data = np.column_stack((x, y, valid_ranges))
            self.apply_distance_filter()
            
    def apply_distance_filter(self):
        """Apply distance filtering to the raw scan data"""
        if self.raw_scan_data is not None:
            # Filter based on distance
            distances = self.raw_scan_data[:, 2]
            mask = (distances >= self.min_distance) & (distances <= self.max_distance)
            self.scan_data = self.raw_scan_data[mask, :2]  # Only keep x, y
            
    def set_distance_filter(self, min_dist, max_dist):
        """Update distance filter parameters"""
        self.min_distance = min_dist
        self.max_distance = max_dist
        self.apply_distance_filter()


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas widget for PyQt5"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8))
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('LiDAR Calibration - Align Virtual Markers with Scan Data')
        
        # Enable interactive mode
        self.setFocusPolicy(Qt.StrongFocus)
        
    def clear_and_setup(self):
        """Clear the plot and set up axes"""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('LiDAR Calibration - Align Virtual Markers with Scan Data')


class AdvancedMarkerCalibrator(QMainWindow):
    """Main GUI application for advanced LiDAR calibration with multiple markers"""
    
    update_plot_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Initialize ROS2
        rclpy.init()
        self.ros_node = LaserScanSubscriber()
        
        # Start ROS2 spinner in a separate thread
        self.ros_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()
        
        # Transformation matrix (identity initially)
        self.T_cumulative = np.eye(3)
        
        # Default values
        self.move_step = 0.01  # meters
        self.angle_step = 1.0  # degrees
        
        # Marker list
        self.markers: List[Marker] = []
        self.current_marker_index: Optional[int] = None
        
        # Initialize UI
        self.init_ui()
        
        # Connect signal for thread-safe plot updates
        self.update_plot_signal.connect(self.update_plot)
        
        # Timer for periodic plot updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot_signal.emit)
        
        # Add default marker
        self.add_marker()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Advanced LiDAR Marker Calibrator')
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel container
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        
        # Main Control & Status Panel
        control_group = QGroupBox("Main Control & Status")
        control_layout = QGridLayout()
        
        # ROS topic
        control_layout.addWidget(QLabel("ROS2 Topic:"), 0, 0)
        self.topic_input = QLineEdit("/scan")
        control_layout.addWidget(self.topic_input, 0, 1)
        
        self.start_button = QPushButton("Start/Connect")
        self.start_button.clicked.connect(self.start_visualization)
        control_layout.addWidget(self.start_button, 0, 2)
        
        # Step controls
        control_layout.addWidget(QLabel("Move Step (m):"), 1, 0)
        self.move_step_input = QLineEdit("0.01")
        control_layout.addWidget(self.move_step_input, 1, 1)
        
        control_layout.addWidget(QLabel("Angle Step (deg):"), 2, 0)
        self.angle_step_input = QLineEdit("1.0")
        control_layout.addWidget(self.angle_step_input, 2, 1)
        
        self.apply_steps_button = QPushButton("Apply Steps")
        self.apply_steps_button.clicked.connect(self.apply_step_size)
        control_layout.addWidget(self.apply_steps_button, 1, 2, 2, 1)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # Current Estimated Extrinsic
        extrinsic_group = QGroupBox("Current Estimated Extrinsic")
        extrinsic_layout = QVBoxLayout()
        self.extrinsic_display = QTextEdit()
        self.extrinsic_display.setReadOnly(True)
        self.extrinsic_display.setMaximumHeight(80)
        extrinsic_layout.addWidget(self.extrinsic_display)
        extrinsic_group.setLayout(extrinsic_layout)
        left_layout.addWidget(extrinsic_group)
        
        # Data Filtering Panel
        filter_group = QGroupBox("Data Filtering")
        filter_layout = QGridLayout()
        
        filter_layout.addWidget(QLabel("Min Distance (m):"), 0, 0)
        self.min_dist_input = QLineEdit("0.1")
        filter_layout.addWidget(self.min_dist_input, 0, 1)
        
        filter_layout.addWidget(QLabel("Max Distance (m):"), 1, 0)
        self.max_dist_input = QLineEdit("10.0")
        filter_layout.addWidget(self.max_dist_input, 1, 1)
        
        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.clicked.connect(self.apply_distance_filter)
        filter_layout.addWidget(self.apply_filter_button, 2, 0, 1, 2)
        
        filter_group.setLayout(filter_layout)
        left_layout.addWidget(filter_group)
        
        # Marker Management Panel
        marker_mgmt_group = QGroupBox("Marker Management")
        marker_mgmt_layout = QVBoxLayout()
        
        # List widget for markers
        self.marker_list = QListWidget()
        self.marker_list.currentRowChanged.connect(self.on_marker_selection_changed)
        marker_mgmt_layout.addWidget(self.marker_list)
        
        # Add/Remove buttons
        marker_buttons_layout = QHBoxLayout()
        self.add_marker_button = QPushButton("+")
        self.add_marker_button.clicked.connect(self.add_marker)
        self.remove_marker_button = QPushButton("-")
        self.remove_marker_button.clicked.connect(self.remove_marker)
        marker_buttons_layout.addWidget(self.add_marker_button)
        marker_buttons_layout.addWidget(self.remove_marker_button)
        marker_mgmt_layout.addLayout(marker_buttons_layout)
        
        marker_mgmt_group.setLayout(marker_mgmt_layout)
        left_layout.addWidget(marker_mgmt_group)
        
        # Marker Properties Panel
        self.properties_group = QGroupBox("Marker Properties")
        properties_layout = QGridLayout()
        
        # Marker pose
        properties_layout.addWidget(QLabel("Marker X (m):"), 0, 0)
        self.marker_x_input = QLineEdit()
        self.marker_x_input.textChanged.connect(self.on_marker_property_changed)
        properties_layout.addWidget(self.marker_x_input, 0, 1)
        
        properties_layout.addWidget(QLabel("Marker Y (m):"), 1, 0)
        self.marker_y_input = QLineEdit()
        self.marker_y_input.textChanged.connect(self.on_marker_property_changed)
        properties_layout.addWidget(self.marker_y_input, 1, 1)
        
        properties_layout.addWidget(QLabel("Marker Theta (deg):"), 2, 0)
        self.marker_theta_input = QLineEdit()
        self.marker_theta_input.textChanged.connect(self.on_marker_property_changed)
        properties_layout.addWidget(self.marker_theta_input, 2, 1)
        
        # Marker dimensions
        properties_layout.addWidget(QLabel("Segment 1 Length (m):"), 3, 0)
        self.segment1_input = QLineEdit()
        self.segment1_input.textChanged.connect(self.on_marker_property_changed)
        properties_layout.addWidget(self.segment1_input, 3, 1)
        
        properties_layout.addWidget(QLabel("Segment 2 Length (m):"), 4, 0)
        self.segment2_input = QLineEdit()
        self.segment2_input.textChanged.connect(self.on_marker_property_changed)
        properties_layout.addWidget(self.segment2_input, 4, 1)
        
        self.properties_group.setLayout(properties_layout)
        self.properties_group.setEnabled(False)
        left_layout.addWidget(self.properties_group)
        
        # Instructions
        instructions_group = QGroupBox("Keyboard Controls")
        instructions_layout = QVBoxLayout()
        instructions_text = QTextEdit()
        instructions_text.setReadOnly(True)
        instructions_text.setMaximumHeight(120)
        instructions_text.setPlainText(
            "Arrow Keys: Translate all markers\n"
            "1: Rotate clockwise\n"
            "2: Rotate counter-clockwise\n\n"
            "Align all virtual markers with the real scan data.\n"
            "The transformation you apply is the inverse of the LiDAR extrinsic."
        )
        instructions_layout.addWidget(instructions_text)
        instructions_group.setLayout(instructions_layout)
        left_layout.addWidget(instructions_group)
        
        # Calculate button
        self.calculate_button = QPushButton("Calculate Final Extrinsic")
        self.calculate_button.clicked.connect(self.calculate_final_extrinsic)
        left_layout.addWidget(self.calculate_button)
        
        left_layout.addStretch()
        
        # Right panel (visualization)
        self.canvas = MatplotlibCanvas()
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add panels to main layout
        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(self.canvas, 2)
        
        # Update initial feedback
        self.update_feedback()
        
    def ros_spin(self):
        """ROS2 spinning thread"""
        rclpy.spin(self.ros_node)
        
    def add_marker(self):
        """Add a new marker to the list"""
        marker_num = len(self.markers) + 1
        new_marker = Marker(name=f"Marker {marker_num}")
        self.markers.append(new_marker)
        
        # Add to list widget
        self.marker_list.addItem(new_marker.name)
        
        # Select the new marker
        self.marker_list.setCurrentRow(len(self.markers) - 1)
        
        # Update plot
        self.update_plot_signal.emit()
        
    def remove_marker(self):
        """Remove the currently selected marker"""
        current_row = self.marker_list.currentRow()
        if current_row >= 0 and len(self.markers) > 1:  # Keep at least one marker
            self.markers.pop(current_row)
            self.marker_list.takeItem(current_row)
            
            # Update selection
            if current_row >= len(self.markers):
                current_row = len(self.markers) - 1
            self.marker_list.setCurrentRow(current_row)
            
            # Update plot
            self.update_plot_signal.emit()
            
    def on_marker_selection_changed(self, current_row):
        """Handle marker selection change"""
        if current_row >= 0 and current_row < len(self.markers):
            self.current_marker_index = current_row
            self.load_marker_properties(self.markers[current_row])
            self.properties_group.setEnabled(True)
        else:
            self.current_marker_index = None
            self.properties_group.setEnabled(False)
            
    def load_marker_properties(self, marker: Marker):
        """Load marker properties into the input fields"""
        # Temporarily disconnect signals to avoid triggering updates
        self.marker_x_input.blockSignals(True)
        self.marker_y_input.blockSignals(True)
        self.marker_theta_input.blockSignals(True)
        self.segment1_input.blockSignals(True)
        self.segment2_input.blockSignals(True)
        
        self.marker_x_input.setText(str(marker.x))
        self.marker_y_input.setText(str(marker.y))
        self.marker_theta_input.setText(str(marker.theta * 180 / np.pi))
        self.segment1_input.setText(str(marker.segment1_length))
        self.segment2_input.setText(str(marker.segment2_length))
        
        # Reconnect signals
        self.marker_x_input.blockSignals(False)
        self.marker_y_input.blockSignals(False)
        self.marker_theta_input.blockSignals(False)
        self.segment1_input.blockSignals(False)
        self.segment2_input.blockSignals(False)
        
    def on_marker_property_changed(self):
        """Handle changes to marker properties"""
        if self.current_marker_index is not None:
            try:
                marker = self.markers[self.current_marker_index]
                marker.x = float(self.marker_x_input.text())
                marker.y = float(self.marker_y_input.text())
                marker.theta = float(self.marker_theta_input.text()) * np.pi / 180
                marker.segment1_length = float(self.segment1_input.text())
                marker.segment2_length = float(self.segment2_input.text())
                
                # Update plot
                self.update_plot_signal.emit()
            except ValueError:
                pass  # Ignore invalid inputs
                
    def start_visualization(self):
        """Start the visualization with current parameters"""
        try:
            # Start ROS subscription
            topic_name = self.topic_input.text()
            self.ros_node.start_subscription(topic_name)
            
            # Start update timer
            self.update_timer.start(100)  # Update every 100ms
            
            # Set focus to canvas for keyboard events
            self.canvas.setFocus()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start visualization: {e}")
            
    def apply_step_size(self):
        """Apply the step size from input fields"""
        try:
            self.move_step = float(self.move_step_input.text())
            self.angle_step = float(self.angle_step_input.text())
            print(f"Step sizes updated: Move={self.move_step}m, Angle={self.angle_step}°")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for step sizes.")
            
    def apply_distance_filter(self):
        """Apply distance filter to the laser scan data"""
        try:
            min_dist = float(self.min_dist_input.text())
            max_dist = float(self.max_dist_input.text())
            
            if min_dist >= max_dist:
                QMessageBox.warning(self, "Invalid Filter", "Min distance must be less than max distance.")
                return
                
            self.ros_node.set_distance_filter(min_dist, max_dist)
            self.update_plot_signal.emit()
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for distances.")
            
    def on_key_press(self, event):
        """Handle keyboard events for transformation"""
        if event.key == 'up':
            self.translate(0, self.move_step)
        elif event.key == 'down':
            self.translate(0, -self.move_step)
        elif event.key == 'left':
            self.translate(-self.move_step, 0)
        elif event.key == 'right':
            self.translate(self.move_step, 0)
        elif event.key == '1':
            self.rotate(-self.angle_step * np.pi / 180)  # Clockwise
        elif event.key == '2':
            self.rotate(self.angle_step * np.pi / 180)   # Counter-clockwise
            
        self.update_feedback()
        self.update_plot_signal.emit()
        
    def translate(self, dx, dy):
        """Apply translation to cumulative transformation"""
        T_translate = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])
        self.T_cumulative = T_translate @ self.T_cumulative
        
    def rotate(self, dtheta):
        """Apply rotation to cumulative transformation"""
        c, s = np.cos(dtheta), np.sin(dtheta)
        T_rotate = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        self.T_cumulative = T_rotate @ self.T_cumulative
        
    def transform_points(self, points):
        """Transform 2D points using the cumulative transformation matrix"""
        # Convert to homogeneous coordinates
        num_points = points.shape[0]
        points_h = np.ones((3, num_points))
        points_h[:2, :] = points.T
        
        # Apply transformation
        transformed_h = self.T_cumulative @ points_h
        
        # Convert back to 2D
        return transformed_h[:2, :].T
        
    def create_l_marker(self, marker: Marker):
        """Create L-shaped marker points in base_link frame"""
        # Transform from base_link to marker
        c, s = np.cos(marker.theta), np.sin(marker.theta)
        T_base_to_marker = np.array([
            [c, -s, marker.x],
            [s, c, marker.y],
            [0, 0, 1]
        ])
        
        # Define L-shape points in marker frame
        marker_points = np.array([
            [0, 0],                          # Origin
            [marker.segment1_length, 0],     # End of segment 1
            [0, 0],                          # Back to origin
            [0, marker.segment2_length]      # End of segment 2
        ])
        
        # Transform to base_link frame
        marker_points_h = np.ones((3, marker_points.shape[0]))
        marker_points_h[:2, :] = marker_points.T
        
        base_points_h = T_base_to_marker @ marker_points_h
        return base_points_h[:2, :].T
        
    def update_plot(self):
        """Update the matplotlib plot"""
        self.canvas.clear_and_setup()
        
        # Plot laser scan data if available
        if self.ros_node.scan_data is not None:
            self.canvas.ax.scatter(self.ros_node.scan_data[:, 0], 
                                 self.ros_node.scan_data[:, 1], 
                                 c='blue', s=1, alpha=0.5, label='Laser Scan')
        
        # Plot all markers
        all_marker_points = []
        for i, marker in enumerate(self.markers):
            marker_points = self.create_l_marker(marker)
            transformed_marker = self.transform_points(marker_points)
            
            # Use different colors for different markers
            color = plt.cm.tab10(i % 10)
            self.canvas.ax.plot(transformed_marker[:, 0], 
                              transformed_marker[:, 1], 
                              '-', linewidth=3, color=color, label=marker.name)
            
            all_marker_points.append(transformed_marker)
        
        # Plot base_link position (transformed)
        base_link = self.transform_points(np.array([[0, 0]]))
        self.canvas.ax.plot(base_link[0, 0], base_link[0, 1], 
                          'go', markersize=10, label='Virtual Base Link')
        
        # Set axis limits
        if self.ros_node.scan_data is not None or all_marker_points:
            all_points = [base_link]
            if self.ros_node.scan_data is not None:
                all_points.append(self.ros_node.scan_data)
            all_points.extend(all_marker_points)
            
            all_points = np.vstack(all_points)
            margin = 0.5
            self.canvas.ax.set_xlim(all_points[:, 0].min() - margin, 
                                   all_points[:, 0].max() + margin)
            self.canvas.ax.set_ylim(all_points[:, 1].min() - margin, 
                                   all_points[:, 1].max() + margin)
        else:
            self.canvas.ax.set_xlim(-2, 2)
            self.canvas.ax.set_ylim(-2, 2)
            
        self.canvas.ax.legend(loc='upper right')
        self.canvas.draw()
        
    def update_feedback(self):
        """Update the live feedback display"""
        # Calculate current estimated extrinsic (inverse of cumulative)
        T_inv = np.linalg.inv(self.T_cumulative)
        x = T_inv[0, 2]
        y = T_inv[1, 2]
        theta_rad = np.arctan2(T_inv[1, 0], T_inv[0, 0])
        theta_deg = theta_rad * 180 / np.pi
        
        feedback_text = f"X: {x:.4f} m\nY: {y:.4f} m\nTheta: {theta_deg:.2f}°"
        self.extrinsic_display.setPlainText(feedback_text)
        
    def calculate_final_extrinsic(self):
        """Calculate and display the final extrinsic calibration"""
        # The extrinsic is the inverse of the cumulative transformation
        T_extrinsic = np.linalg.inv(self.T_cumulative)
        
        x = T_extrinsic[0, 2]
        y = T_extrinsic[1, 2]
        theta_rad = np.arctan2(T_extrinsic[1, 0], T_extrinsic[0, 0])
        theta_deg = theta_rad * 180 / np.pi
        
        result = f"\n{'='*50}\n"
        result += "FINAL LIDAR EXTRINSIC CALIBRATION RESULT:\n"
        result += f"{'='*50}\n"
        result += f"X: {x:.6f} m\n"
        result += f"Y: {y:.6f} m\n"
        result += f"Theta: {theta_deg:.4f}° ({theta_rad:.6f} rad)\n"
        result += f"{'='*50}\n"
        
        print(result)
        
        # Show in a message box
        QMessageBox.information(self, "Calibration Result", result)
        
    def closeEvent(self, event):
        """Handle window close event"""
        self.update_timer.stop()
        
        # Calculate and print final extrinsic
        T_extrinsic = np.linalg.inv(self.T_cumulative)
        x = T_extrinsic[0, 2]
        y = T_extrinsic[1, 2]
        theta_rad = np.arctan2(T_extrinsic[1, 0], T_extrinsic[0, 0])
        theta_deg = theta_rad * 180 / np.pi
        
        print(f"\nFinal LiDAR Extrinsic: x={x:.6f}m, y={y:.6f}m, theta={theta_deg:.4f}°")
        
        rclpy.shutdown()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    calibrator = AdvancedMarkerCalibrator()
    calibrator.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
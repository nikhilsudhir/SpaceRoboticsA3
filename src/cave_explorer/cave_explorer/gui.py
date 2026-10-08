#!/usr/bin/env python3
"""
Mission-control GUI for cave_explorer: a live mission-status dashboard, separate from RViz.

Shows the current mode/elapsed time/artefact counts/average speed, a mini schematic map
(robot, current goal, artefact pins), a live artefact inventory table, and a scrolling event
timeline - all driven by the 'gui_status' JSON topic published by cave_explorer_node (see
publish_gui_status() there). Controls: Start Exploring, Pause/Resume, Force Return Home, and
Export Summary (writes a text report for pasting into the project report).

Run with: ros2 run cave_explorer gui
(after the other three launch files, same as RViz - see README).
"""

import json
import sys
from datetime import datetime

import numpy as np
import rclpy
from PyQt5 import QtCore, QtGui, QtWidgets
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

# How often (ms) the GUI polls ROS and refreshes widgets. Independent of how often
# cave_explorer_node actually publishes (GUI_STATUS_PERIOD_S there) - this just needs to be
# fast enough that the UI feels live and Trigger service calls get processed promptly.
GUI_POLL_PERIOD_MS = 200

# How often (ms) the manual-override D-pad re-publishes the current cmd_vel while a direction
# button is held - see MainWindow's manual-control section. 10Hz, matching typical teleop rate.
MANUAL_CONTROL_PERIOD_MS = 100
MANUAL_LINEAR_MPS = 0.4   # conservative - well under max_vel_x in nav2_params.yaml
MANUAL_ANGULAR_RPS = 0.6

# Occupancy grid cell colours, matching the usual RViz map look (dark = occupied, light =
# free, teal-grey = unknown/unexplored).
MAP_COLOR_UNKNOWN = (90, 110, 120)
MAP_COLOR_FREE = (225, 225, 225)
MAP_COLOR_OCCUPIED = (20, 20, 25)

STATUS_COLORS = {
    'visited': '#2ecc71',
    'abandoned': '#e67e22',
    'pending': '#e74c3c',
    'detected': '#8d99ae',
}

# What verb to show in the timeline the first time an artefact is seen in each status -
# "pending"/"detected" are both first-sightings (just distinguishing inspectable vs not).
TIMELINE_VERBS = {
    'detected': 'FOUND',
    'pending': 'FOUND',
    'visited': 'INSPECTED',
    'abandoned': 'ABANDONED',
}

STYLESHEET = """
QWidget {
    background-color: #1e2127;
    color: #e6e6e6;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel#header {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
    padding: 4px 0px;
}
QGroupBox {
    background-color: #272b33;
    border: 1px solid #3a3f4b;
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px;
    font-weight: 600;
    color: #9fb3c8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0px 6px;
}
QLabel.value {
    color: #ffffff;
    font-weight: 600;
}
QPushButton {
    background-color: #3a3f4b;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton:hover:!disabled {
    background-color: #4b5160;
}
QPushButton:disabled {
    background-color: #2d313a;
    color: #6c7280;
}
QPushButton#startButton {
    background-color: #2ecc71;
    color: #10331f;
    font-size: 15px;
}
QPushButton#startButton:hover:!disabled {
    background-color: #3fd87f;
}
QPushButton#startButton:disabled {
    background-color: #2d313a;
    color: #6c7280;
}
QTableWidget {
    background-color: #1e2127;
    gridline-color: #3a3f4b;
    border: none;
    selection-background-color: #3a3f4b;
}
QHeaderView::section {
    background-color: #272b33;
    color: #9fb3c8;
    padding: 6px;
    border: none;
    font-weight: 600;
}
QListWidget {
    background-color: #1e2127;
    border: 1px solid #3a3f4b;
    border-radius: 6px;
    font-family: "Consolas", monospace;
    font-size: 12px;
}
"""


class GuiRosNode(Node):
    """Thin ROS2 node the GUI drives directly from a QTimer - see MainWindow.poll()."""

    def __init__(self):
        super().__init__('cave_explorer_gui')
        self.latest_status = None
        self.create_subscription(String, 'gui_status', self._status_callback, 10)
        self.start_mission_client = self.create_client(Trigger, 'start_mission')
        self.pause_mission_client = self.create_client(Trigger, 'pause_mission')
        self.resume_mission_client = self.create_client(Trigger, 'resume_mission')
        self.force_return_home_client = self.create_client(Trigger, 'force_return_home')

        # Real occupancy-grid map for MapWidget's background - subscribed directly here
        # (not routed through the gui_status JSON topic, which would be wasteful for a whole
        # grid) and converted to a QImage once per map update rather than once per poll tick.
        self.latest_map_image = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.create_subscription(OccupancyGrid, 'map', self._map_callback, 1)

        # Manual override (see MainWindow's Manual Override panel) - only ever actually
        # published while the mission is paused or not yet started (enforced in MainWindow,
        # not here), so it can never fight with Nav2's own cmd_vel commands.
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self._manual_twist = Twist()

    def _status_callback(self, msg):
        try:
            self.latest_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Received malformed gui_status message')

    def _map_callback(self, msg):
        width, height = msg.info.width, msg.info.height
        if width == 0 or height == 0:
            return

        grid = np.array(msg.data, dtype=np.int8).reshape(height, width)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[grid < 0] = MAP_COLOR_UNKNOWN
        rgb[(grid >= 0) & (grid < 50)] = MAP_COLOR_FREE
        rgb[grid >= 50] = MAP_COLOR_OCCUPIED

        # OccupancyGrid row 0 is at the origin (bottom in world-space); QImage row 0 is the
        # top of the image, so flip vertically to display it the right way up.
        rgb = np.ascontiguousarray(np.flipud(rgb))
        image = QtGui.QImage(rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
        self.latest_map_image = image.copy()  # copy - rgb's buffer goes out of scope
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

    def set_manual_twist(self, linear_x, angular_z):
        self._manual_twist.linear.x = linear_x
        self._manual_twist.angular.z = angular_z

    def publish_manual_twist(self):
        self.cmd_vel_pub.publish(self._manual_twist)

    def _call(self, client, name):
        if not client.service_is_ready():
            self.get_logger().warn(f'{name} service not available yet - is cave_explorer_node running?')
            return False
        client.call_async(Trigger.Request())
        return True

    def call_start_mission(self):
        return self._call(self.start_mission_client, 'start_mission')

    def call_pause_mission(self):
        return self._call(self.pause_mission_client, 'pause_mission')

    def call_resume_mission(self):
        return self._call(self.resume_mission_client, 'resume_mission')

    def call_force_return_home(self):
        return self._call(self.force_return_home_client, 'force_return_home')


class MapWidget(QtWidgets.QWidget):
    """
    The real occupancy-grid map (as built by SLAM - same data RViz's Map display shows),
    with robot position, current goal, and artefact pins overlaid on top.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._map_image = None
        self._resolution = None
        self._origin_x = None
        self._origin_y = None
        self._robot = None
        self._goal = None
        self._artifacts = []

    def update_data(self, status, map_image, resolution, origin_x, origin_y):
        self._map_image = map_image
        self._resolution = resolution
        self._origin_x = origin_x
        self._origin_y = origin_y
        rx, ry = status.get('robot_x'), status.get('robot_y')
        self._robot = (rx, ry) if rx is not None and ry is not None else None
        gx, gy = status.get('goal_x'), status.get('goal_y')
        self._goal = (gx, gy) if gx is not None and gy is not None else None
        self._artifacts = status.get('artifacts', [])
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = QtCore.QRectF(self.rect().adjusted(6, 6, -6, -6))

        painter.fillRect(self.rect(), QtGui.QColor('#14161a'))
        painter.setPen(QtGui.QPen(QtGui.QColor('#3a3f4b'), 1))
        painter.drawRoundedRect(rect, 6, 6)

        if self._map_image is None:
            painter.setPen(QtGui.QColor('#6c7280'))
            painter.drawText(rect, QtCore.Qt.AlignCenter, 'Waiting for map...')
            return

        # Draw the occupancy grid scaled to fit the widget, preserving its aspect ratio
        # (fitting by the smaller scale factor and centring, rather than stretching it).
        img_w, img_h = self._map_image.width(), self._map_image.height()
        scale = min(rect.width() / img_w, rect.height() / img_h)
        draw_w, draw_h = img_w * scale, img_h * scale
        offset_x = rect.left() + (rect.width() - draw_w) / 2
        offset_y = rect.top() + (rect.height() - draw_h) / 2
        painter.drawImage(QtCore.QRectF(offset_x, offset_y, draw_w, draw_h), self._map_image)

        def world_to_widget(x, y):
            col = (x - self._origin_x) / self._resolution
            row = (y - self._origin_y) / self._resolution
            image_row = img_h - 1 - row  # the image was flipped vertically when built - see _map_callback()
            return QtCore.QPointF(offset_x + col * scale, offset_y + image_row * scale)

        for art in self._artifacts:
            point = world_to_widget(art['x'], art['y'])
            painter.setBrush(QtGui.QColor(STATUS_COLORS.get(art['status'], '#ffffff')))
            painter.setPen(QtGui.QPen(QtGui.QColor('#000000'), 0.5))
            painter.drawEllipse(point, 4, 4)

        if self._goal is not None:
            point = world_to_widget(self._goal[0], self._goal[1])
            painter.setPen(QtGui.QPen(QtGui.QColor('#f1c40f'), 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(point, 7, 7)

        if self._robot is not None:
            point = world_to_widget(self._robot[0], self._robot[1])
            painter.setBrush(QtGui.QColor('#3498db'))
            painter.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), 1))
            painter.drawEllipse(point, 6, 6)


class MainWindow(QtWidgets.QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        # (artefact_id, status) pairs already shown in the timeline, so each transition
        # (discovered -> inspected/abandoned) is only announced once, not every poll tick.
        self._announced_events = set()
        self._mission_complete_announced = False
        self._manual_control_enabled = False

        self.setWindowTitle('Cave Explorer - Mission Control')
        self.resize(980, 720)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start(GUI_POLL_PERIOD_MS)

        # Separate, faster timer purely for re-publishing the manual-override cmd_vel while a
        # D-pad button is held (see _build_manual_control_box()) - kept independent of
        # poll_timer's slower cadence so holding a direction key feels responsive.
        self.manual_timer = QtCore.QTimer(self)
        self.manual_timer.timeout.connect(self._publish_manual_velocity_if_enabled)
        self.manual_timer.start(MANUAL_CONTROL_PERIOD_MS)

    def _publish_manual_velocity_if_enabled(self):
        if self._manual_control_enabled:
            self.ros_node.publish_manual_twist()

    def _update_manual_control_enabled(self, status):
        # Safe whenever main_loop() isn't actively driving: before start, while paused, or
        # once the mission has genuinely finished (at which point the Pause button is itself
        # disabled, so this would otherwise be the one state with no way to unlock it).
        should_enable = (not status['mission_started']) or status['mission_paused'] \
            or status['mission_complete']
        if should_enable == self._manual_control_enabled:
            return

        self._manual_control_enabled = should_enable
        for button in self.manual_buttons:
            button.setEnabled(should_enable)
        if should_enable:
            self.manual_hint.setText('Manual control active - use the D-pad to drive directly')
        else:
            # Transitioning OUT of manual mode (e.g. mission resumed) - force an immediate
            # stop rather than leaving a stale non-zero twist from whatever was last held.
            self._set_manual_velocity(0.0, 0.0)
            self.manual_hint.setText('Pause the mission to enable manual control')

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(10)

        header = QtWidgets.QLabel('Cave Explorer — Mission Control')
        header.setObjectName('header')
        root.addWidget(header)

        # --- Controls row ---
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(10)
        self.start_button = QtWidgets.QPushButton('Start Exploring')
        self.start_button.setObjectName('startButton')
        self.start_button.clicked.connect(self.on_start_clicked)
        self.pause_button = QtWidgets.QPushButton('Pause')
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        self.return_home_button = QtWidgets.QPushButton('Force Return Home')
        self.return_home_button.setEnabled(False)
        self.return_home_button.clicked.connect(self.on_return_home_clicked)
        self.export_button = QtWidgets.QPushButton('Export Summary')
        self.export_button.clicked.connect(self.on_export_clicked)
        controls.addWidget(self.start_button, 2)
        controls.addWidget(self.pause_button, 1)
        controls.addWidget(self.return_home_button, 1)
        controls.addWidget(self.export_button, 1)
        root.addLayout(controls)

        # --- Two-column body ---
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        body.addLayout(self._build_left_column(), 2)
        body.addLayout(self._build_right_column(), 3)
        root.addLayout(body)

    def _build_left_column(self):
        column = QtWidgets.QVBoxLayout()
        column.setSpacing(12)

        status_box = QtWidgets.QGroupBox('Mission Status')
        status_layout = QtWidgets.QFormLayout()
        status_layout.setSpacing(8)
        self.mode_label = self._value_label('WAITING')
        self.elapsed_label = self._value_label('0s')
        self.counts_label = self._value_label('visited=0  abandoned=0  pending=0')
        self.speed_label = self._value_label('0.00 m/s')
        status_layout.addRow('Mode:', self.mode_label)
        status_layout.addRow('Elapsed:', self.elapsed_label)
        status_layout.addRow('Artefacts:', self.counts_label)
        status_layout.addRow('Avg speed:', self.speed_label)
        status_box.setLayout(status_layout)
        column.addWidget(status_box)

        map_box = QtWidgets.QGroupBox('Map')
        map_layout = QtWidgets.QVBoxLayout()
        self.map_widget = MapWidget()
        map_layout.addWidget(self.map_widget)
        legend = QtWidgets.QLabel(
            '<span style="color:#3498db">●</span> robot&nbsp;&nbsp;'
            '<span style="color:#f1c40f">○</span> goal&nbsp;&nbsp;'
            '<span style="color:#2ecc71">●</span> visited&nbsp;&nbsp;'
            '<span style="color:#e74c3c">●</span> pending&nbsp;&nbsp;'
            '<span style="color:#e67e22">●</span> abandoned')
        legend.setStyleSheet('color: #9fb3c8; font-size: 11px;')
        map_layout.addWidget(legend)
        map_box.setLayout(map_layout)
        column.addWidget(map_box, 1)

        column.addWidget(self._build_manual_control_box())

        return column

    def _build_manual_control_box(self):
        """
        D-pad for direct teleop via /cmd_vel. Only ever enabled while the mission is paused
        or not yet started (see _update_manual_control_enabled()) - autonomous driving and
        manual driving must never both be sending cmd_vel at once, so this is the one safety
        rule that's non-negotiable here, not just a UI nicety.
        """

        box = QtWidgets.QGroupBox('Manual Override')
        layout = QtWidgets.QVBoxLayout()

        self.manual_hint = QtWidgets.QLabel('Pause the mission to enable manual control')
        self.manual_hint.setStyleSheet('color: #9fb3c8; font-size: 11px;')
        layout.addWidget(self.manual_hint)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)

        def make_button(text):
            button = QtWidgets.QPushButton(text)
            button.setEnabled(False)
            return button

        self.btn_forward = make_button('▲ Forward')
        self.btn_backward = make_button('▼ Backward')
        self.btn_left = make_button('◀ Left')
        self.btn_right = make_button('▶ Right')
        self.btn_stop = make_button('■ Stop')

        grid.addWidget(self.btn_forward, 0, 1)
        grid.addWidget(self.btn_left, 1, 0)
        grid.addWidget(self.btn_stop, 1, 1)
        grid.addWidget(self.btn_right, 1, 2)
        grid.addWidget(self.btn_backward, 2, 1)
        layout.addLayout(grid)
        box.setLayout(layout)

        self.manual_buttons = [self.btn_forward, self.btn_backward, self.btn_left, self.btn_right,
                                self.btn_stop]

        # Hold-to-move: publish while pressed, zero on release. btn_stop is a one-shot safety
        # stop rather than hold-to-move (it has nothing to "hold" for).
        self.btn_forward.pressed.connect(lambda: self._set_manual_velocity(MANUAL_LINEAR_MPS, 0.0))
        self.btn_backward.pressed.connect(lambda: self._set_manual_velocity(-MANUAL_LINEAR_MPS, 0.0))
        self.btn_left.pressed.connect(lambda: self._set_manual_velocity(0.0, MANUAL_ANGULAR_RPS))
        self.btn_right.pressed.connect(lambda: self._set_manual_velocity(0.0, -MANUAL_ANGULAR_RPS))
        for button in self.manual_buttons:
            button.released.connect(lambda: self._set_manual_velocity(0.0, 0.0))
        self.btn_stop.clicked.connect(lambda: self._set_manual_velocity(0.0, 0.0))

        return box

    def _set_manual_velocity(self, linear_x, angular_z):
        self.ros_node.set_manual_twist(linear_x, angular_z)

    def _build_right_column(self):
        column = QtWidgets.QVBoxLayout()
        column.setSpacing(12)

        table_box = QtWidgets.QGroupBox('Artefact Inventory')
        table_layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['ID', 'Label', 'Position', 'Status'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.table)
        table_box.setLayout(table_layout)
        column.addWidget(table_box, 2)

        timeline_box = QtWidgets.QGroupBox('Event Timeline')
        timeline_layout = QtWidgets.QVBoxLayout()
        self.timeline = QtWidgets.QListWidget()
        timeline_layout.addWidget(self.timeline)
        timeline_box.setLayout(timeline_layout)
        column.addWidget(timeline_box, 2)

        return column

    @staticmethod
    def _value_label(text):
        label = QtWidgets.QLabel(text)
        label.setProperty('class', 'value')
        label.setStyleSheet('color: #ffffff; font-weight: 600;')
        return label

    # --- Button handlers ---

    def on_start_clicked(self):
        if self.ros_node.call_start_mission():
            self.start_button.setEnabled(False)
            self.start_button.setText('Exploring...')
            self.pause_button.setEnabled(True)
            self.return_home_button.setEnabled(True)

    def on_pause_clicked(self):
        status = self.ros_node.latest_status
        is_paused = bool(status and status.get('mission_paused'))
        if is_paused:
            self.ros_node.call_resume_mission()
        else:
            self.ros_node.call_pause_mission()

    def on_return_home_clicked(self):
        self.ros_node.call_force_return_home()
        self.return_home_button.setEnabled(False)

    def on_export_clicked(self):
        status = self.ros_node.latest_status
        if status is None:
            QtWidgets.QMessageBox.information(self, 'Export Summary', 'No status received yet.')
            return

        default_name = f"cave_explorer_summary_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Run Summary', default_name,
                                                          'Text files (*.txt)')
        if not path:
            return

        lines = [
            'Cave Explorer - Run Summary',
            f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',
            '',
            f"Mode:        {status['mode']}",
            f"Elapsed:     {status['elapsed_s']:.0f}s",
            f"Visited:     {status['visited']}",
            f"Abandoned:   {status['abandoned']}",
            f"Pending:     {status['pending']}",
            f"Avg speed:   {status['avg_speed_mps']:.2f} m/s",
            '',
            'Artefact Inventory:',
        ]
        for art in sorted(status['artifacts'], key=lambda a: a['id']):
            lines.append(
                f"  #{art['id']:<4} {art['label']:<16} ({art['x']:>7.2f}, {art['y']:>7.2f})  {art['status']}")

        try:
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, 'Export Summary', f'Could not save file:\n{e}')
            return

        QtWidgets.QMessageBox.information(self, 'Export Summary', f'Saved to:\n{path}')

    # --- Live polling ---

    def poll(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0)
        status = self.ros_node.latest_status
        if status is None:
            return

        self.mode_label.setText(status['mode'])
        self.elapsed_label.setText(f"{status['elapsed_s']:.0f}s")
        self.counts_label.setText(
            f"visited={status['visited']}  abandoned={status['abandoned']}  "
            f"pending={status['pending']}")
        self.speed_label.setText(f"{status['avg_speed_mps']:.2f} m/s")
        self.map_widget.update_data(
            status, self.ros_node.latest_map_image, self.ros_node.map_resolution,
            self.ros_node.map_origin_x, self.ros_node.map_origin_y)
        self._update_manual_control_enabled(status)

        if status['mission_started']:
            if self.start_button.isEnabled():
                self.start_button.setEnabled(False)
                self.start_button.setText('Exploring...')
            self.pause_button.setEnabled(not status['mission_complete'])
            self.return_home_button.setEnabled(not status['mission_complete'])
            self.pause_button.setText('Resume' if status['mission_paused'] else 'Pause')

        if status['mission_complete']:
            self.start_button.setText('Mission Complete')
            if not self._mission_complete_announced:
                self._mission_complete_announced = True
                self._add_timeline_event(
                    f"MISSION COMPLETE - {status['visited']} inspected, "
                    f"{status['abandoned']} abandoned, {status['elapsed_s']:.0f}s total, "
                    f"avg {status['avg_speed_mps']:.2f} m/s")

        self._refresh_table(status['artifacts'])
        self._refresh_timeline(status['artifacts'])

    def _refresh_table(self, artifacts):
        self.table.setRowCount(len(artifacts))
        for row, art in enumerate(sorted(artifacts, key=lambda a: a['id'])):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(art['id'])))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(art['label']))
            self.table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(f"({art['x']:.1f}, {art['y']:.1f})"))
            status_item = QtWidgets.QTableWidgetItem(art['status'])
            status_item.setForeground(QtGui.QColor('#14161a'))
            status_item.setBackground(QtGui.QColor(STATUS_COLORS.get(art['status'], '#ffffff')))
            self.table.setItem(row, 3, status_item)

    def _refresh_timeline(self, artifacts):
        for art in sorted(artifacts, key=lambda a: a['id']):
            key = (art['id'], art['status'])
            if key in self._announced_events:
                continue
            self._announced_events.add(key)
            verb = TIMELINE_VERBS[art['status']]
            self._add_timeline_event(
                f"{verb}: {art['label']} #{art['id']} at ({art['x']:.1f}, {art['y']:.1f})")

    def _add_timeline_event(self, text):
        self.timeline.addItem(text)
        self.timeline.scrollToBottom()


def main():
    rclpy.init()
    ros_node = GuiRosNode()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(ros_node)
    window.show()
    app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from morai_msgs.msg import CtrlCmd
from obstacle_detector.msg import Obstacles
from std_msgs.msg import Header
from sklearn.cluster import KMeans
from math import cos, sin, pi, sqrt, atan2, tan, radians
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sklearn.linear_model import RANSACRegressor

class Rubber_cone:
    def __init__(self):
        rospy.init_node('rubber_cone')
        rospy.Subscriber("/raw_obstacles", Obstacles, self.obstacle_callback)
        self.is_obstacles = False
        self.obstacles = []

        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd', CtrlCmd, queue_size=1)
        self.cluster_centers_pub = rospy.Publisher('/kmeans_cluster_centers', PoseArray, queue_size=1)
        self.path_pub = rospy.Publisher('/center_path', Path, queue_size=1)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2
        self.ctrl_cmd_msg.velocity = 7.0
        self.ctrl_cmd_msg.accel = 0.1
        self.ctrl_cmd_msg.steering = 0.0

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.is_obstacles:
                self.process_obstacles()
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                self.is_obstacles = False
            rate.sleep()

    def obstacle_callback(self, msg):
        self.is_obstacles = True
        self.obstacles = msg.circles

    def process_obstacles(self):
        if not self.obstacles:
            return

        data = np.array([[obstacle.center.x, obstacle.center.y] for obstacle in self.obstacles])
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
        #labels = kmeans.labels_
        #centers = kmeans.cluster_centers_
        #print(kmeans.labels_)

        left_data = data[kmeans.labels_ == 0]
        right_data = data[kmeans.labels_ == 1]

        # min_samples 값을 동적으로 설정
        min_samples_left = max(2, int(0.2 * len(left_data)))  # 최소 2개, 또는 데이터의 20%
        min_samples_right = max(2, int(0.2 * len(right_data)))

        # RANSAC 회귀 모델 생성 및 데이터 적합성 검사
        if len(left_data) >= min_samples_left:
            ransac_left = RANSACRegressor(min_samples=min_samples_left)
            ransac_left.fit(left_data[:, 0].reshape(-1, 1), left_data[:, 1])

        if len(right_data) >= min_samples_right:
            ransac_right = RANSACRegressor(min_samples=min_samples_right)
            ransac_right.fit(right_data[:, 0].reshape(-1, 1), right_data[:, 1])

        # 경로 추정을 위한 x 값 범위 설정 및 중앙 경로 계산
        x_vals = np.linspace(min(data[:, 0]), max(data[:, 0]), 300).reshape(-1, 1)

        if len(left_data) >= min_samples_left and len(right_data) >= min_samples_right:
            left_y_vals = ransac_left.predict(x_vals)
            right_y_vals = ransac_right.predict(x_vals)
            center_y_vals = (left_y_vals + right_y_vals) / 2

            path_msg = Path()
            path_msg.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')

            for i in range(len(x_vals)):
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = x_vals[i][0]
                pose.pose.position.y = center_y_vals[i]
                pose.pose.position.z = 0
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)
            print("Center path published.")

if __name__ == '__main__':
    try:
        rubber_cone = Rubber_cone()
    except rospy.ROSInternalException:
        pass

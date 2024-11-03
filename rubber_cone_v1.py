#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from morai_msgs.msg import CtrlCmd
from obstacle_detector.msg import Obstacles
from math import cos, sin, pi, sqrt, atan2, tan, radians


class Rubber_cone:
    def __init__(self):
        rospy.init_node('rubber_cone')
        rospy.Subscriber("/raw_obstacles", Obstacles, self.obstacle_callback)
        self.is_obstacles = False
        self.obstacles = []
        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd', CtrlCmd, queue_size=1)
        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 1
        self.ctrl_cmd_msg.velocity = 10.0
        self.ctrl_cmd_msg.accel = 0.1
        self.ctrl_cmd_msg.steering = 0.0
        self.prev_angle = 0.0
        self.robot_direction = np.array([1, 0])  # 전방
        self.avoidance_direction = np.array([1, 0])
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.is_obstacles:

                self.calc_avoidance_direction()
                self.calc_steering_angle()
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                
                self.is_obstacles = False
            rate.sleep()

    def obstacle_callback(self, msg):
        self.is_obstacles=True
        self.obstacles = msg.circles

    def calc_avoidance_direction(self):
        x=0
        y=0
        for obstacle in self.obstacles:
            x += obstacle.center.x
            y += obstacle.center.y

        self.avoidance_direction[0] = x
        self.avoidance_direction[1] = y

    def calc_steering_angle(self):

        # 벡터 사이의 각도 계산
        dot_product = np.dot(self.robot_direction, self.avoidance_direction)
        magnitude_robot = np.linalg.norm(self.robot_direction)
        magnitude_avoidance = np.linalg.norm(self.avoidance_direction)
        
        cos_angle = dot_product / (magnitude_robot * magnitude_avoidance)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 각도를 라디안으로 계산

        if self.avoidance_direction[1] > 0:
            self.ctrl_cmd_msg.steering = angle
        else:
            self.ctrl_cmd_msg.steering = -angle

        print(self.ctrl_cmd_msg.steering*180/pi)


if __name__ == '__main__':
    try:
        rubber_cone = Rubber_cone()
    except rospy.ROSInternalException:
        pass 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from morai_msgs.msg import CtrlCmd
from obstacle_detector.msg import Obstacles
from math import cos, sin, pi, sqrt, atan2, tan, radians

class Vector:
    def __init__(self, m, angle):
        self.m = m
        self.x = None
        self.y = None
        self.d = 10
        self.angle = angle

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
                r = 1
                avoidance_vector_d = 0
                avoidance_vector_angle = 0
                steering_angle = 0

                for degree in range(-45, 46, 1):  # -45도에서 45도까지
                    radian = radians(degree)
                    m = tan(radian)
                    vector = Vector(m, degree)
                    prev_d = 10

                    for obstacle in self.obstacles:
                        A = 1 + m ** 2
                        B = -2 * (obstacle.center.x + m * obstacle.center.y)
                        C = obstacle.center.x ** 2 + obstacle.center.y ** 2 - r ** 2
                        D = B ** 2 - 4 * A * C

                        if D > 0:
                            x1 = (-B + sqrt(D)) / (2 * A)
                            x2 = (-B - sqrt(D)) / (2 * A)
                            x = min(x1, x2)
                            y = m * x
                            d = sqrt(x ** 2 + y ** 2)
                            if d < prev_d:
                                prev_d = d
                                vector.d = d

                    if vector.d > avoidance_vector_d:
                        avoidance_vector_d = vector.d
                        avoidance_vector_angle = vector.angle
                    elif vector.d == avoidance_vector_d:
                        diff_prev_and_vector_angle = abs(self.prev_angle - vector.angle)
                        diff_prev_and_avoidance_vector_angle = abs(self.prev_angle - avoidance_vector_angle)
                        if diff_prev_and_vector_angle < diff_prev_and_avoidance_vector_angle:
                            avoidance_vector_d = vector.d
                            avoidance_vector_angle = vector.angle

                    steering_angle = avoidance_vector_angle

                self.ctrl_cmd_msg.steering = steering_angle*pi/180
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                print(steering_angle)
                self.prev_angle = steering_angle
                self.is_obstacles = False
            rate.sleep()

    def obstacle_callback(self, msg):
        self.is_obstacles=True
        self.obstacles = msg.circles

    def calc_angle_between_vector(self,vector1,vector2):    # 벡터 사이의 각도 계산

        dot_product = np.dot(vector1, vector2)
        magnitude_vector1 = np.linalg.norm(vector1)
        magnitude_vector2 = np.linalg.norm(vector2)
        
        cos_angle = dot_product / (magnitude_vector1 * magnitude_vector2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 각도를 라디안으로 계산

        return angle


if __name__ == '__main__':
    try:
        rubber_cone = Rubber_cone()
    except rospy.ROSInternalException:
        pass 

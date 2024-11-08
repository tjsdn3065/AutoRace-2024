#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from morai_msgs.msg import CtrlCmd
from obstacle_detector.msg import Obstacles
from std_msgs.msg import Header
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor,LinearRegression
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped,Point
from geometry_msgs.msg import PoseArray, Pose
from math import cos,sin,pi,sqrt,pow,atan2,tan

class Rubber_cone:
    def __init__(self):
        rospy.init_node('rubber_cone')
        rospy.Subscriber("/raw_obstacles", Obstacles, self.obstacle_callback)
        self.is_obstacles = False
        self.obstacles = []

        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd', CtrlCmd, queue_size=1)
        self.cluster_centers_pub = rospy.Publisher('/kmeans_cluster_centers', PoseArray, queue_size=1)
        self.path_pub = rospy.Publisher('/center_path', Path, queue_size=1)
        self.left_path_pub = rospy.Publisher('/left_path', Path, queue_size=1)
        self.right_path_pub = rospy.Publisher('/right_path', Path, queue_size=1)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2
        self.ctrl_cmd_msg.velocity = 5.0
        self.ctrl_cmd_msg.accel = 0.1
        self.ctrl_cmd_msg.steering = 0.0

        self.vehicle_length = 1.63
        self.lfd = 2
        self.forward_point = Point()
        self.is_look_forward_point = False


        rate = rospy.Rate(10)  # 10Hz의 주기로 루프 실행

        while not rospy.is_shutdown():
            if self.is_obstacles and len(self.obstacles) >= 2:
                self.process_obstacles()
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                self.is_obstacles = False
            rate.sleep()

    def obstacle_callback(self, msg):
        self.is_obstacles = True
        self.obstacles = msg.circles

    def process_obstacles(self):
        """
        process_obstacles 메소드:
        장애물 데이터를 기반으로 경로를 계산.
        KMeans 클러스터링을 사용하여 장애물 데이터를 두 군집(왼쪽, 오른쪽)으로 분류.
        RANSAC 알고리즘을 이용하여 각 군집 데이터로부터 로봇 경로(왼쪽, 오른쪽, 중앙) 계산.
        각 경로에 대해 ROS 토픽으로 메시지 발행.
        """
        if not self.obstacles:
            return

        data = np.array([[obstacle.center.x, obstacle.center.y] for obstacle in self.obstacles])

        if len(data) < 2:
            print("Not enough data to form clusters.")
            return

        # k-평균 파라미터
        self.valid_clusters = False
        self.attempts = 0
        self.max_attempts = 10  # 최대 시도 횟수 설정

        # 군집을 좌우로 두 개 만들때 각 군집안에 객체가 최소 2개는 있어야함
        # 각 군집의 중심좌표가 0<x<4 만족하면서 왼쪽은 0<y<3, 오른쪽은 -3<y<0을 만족해야함
        while not self.valid_clusters and self.attempts < self.max_attempts:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
            centers = kmeans.cluster_centers_

            # 조건 검사
            if self.validate_clusters(centers):
                self.valid_clusters = True
                self.attempts=0
                print("Valid clusters found:", centers)
                # 군집 중심 좌표를 PoseArray로 발행
                pose_array_msg = PoseArray()
                pose_array_msg.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                for center in kmeans.cluster_centers_:
                    pose = Pose()
                    pose.position.x = center[0]
                    pose.position.y = center[1]
                    pose.position.z = 0
                    pose_array_msg.poses.append(pose)
                self.cluster_centers_pub.publish(pose_array_msg)

                if centers[0][1] < centers[1][1]:
                    left_index, right_index = 1, 0
                else:
                    left_index, right_index = 0, 1

                left_data = data[kmeans.labels_ == left_index]
                right_data = data[kmeans.labels_ == right_index]

                x_vals = np.linspace(min(data[:, 0]), max(data[:, 0]), 30).reshape(-1, 1)

                # 중앙 경로 데이터 초기화
                self.center_path = Path()
                self.center_path.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')

                left_y_vals = fit_and_validate_path(left_data, x_vals, max(2, int(0.2 * len(left_data))), 0.05, 300, 'left')
                right_y_vals = fit_and_validate_path(right_data, x_vals, max(2, int(0.2 * len(right_data))), 0.05, 300, 'right')
                center_y_vals = None

                if left_y_vals is not None and right_y_vals is not None:
                    center_y_vals = calculate_center_path(left_y_vals, right_y_vals, x_vals)
                    publish_path(self.left_path_pub, x_vals, left_y_vals, 'velodyne')
                    publish_path(self.right_path_pub, x_vals, right_y_vals, 'velodyne')
                    publish_path(self.path_pub, x_vals, center_y_vals, 'velodyne')


                    self.center_path.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                    if center_y_vals is not None:
                        for x, y in zip(x_vals, center_y_vals):
                            pose = PoseStamped()
                            pose.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                            pose.pose.position.x = x[0]
                            pose.pose.position.y = y
                            pose.pose.position.z = 0
                            self.center_path.poses.append(pose)

                        theta = 0

                        for i, pose in enumerate(self.center_path.poses):
                            path_point = pose.pose.position
                            dis = sqrt(path_point.x**2 + path_point.y**2)
                            if dis >= self.lfd:
                                self.forward_point = path_point
                                self.is_look_forward_point = True
                                theta = atan2(path_point.y, path_point.x)  # Correct order of arguments
                                break

                        if self.is_look_forward_point:
                            self.ctrl_cmd_msg.steering = atan2(2 * self.vehicle_length * sin(theta), self.lfd)
                            #print(self.ctrl_cmd_msg.steering * 180 / pi)  # Conversion to degrees for readability
                            self.is_look_forward_point = False
            else:
                self.attempts += 1
                print("Retrying... Attempt", self.attempts)
        
        if not self.valid_clusters:
            print("Failed to find valid clusters within the attempts limit.")

    def validate_clusters(self, centers):
        # 군집 중심 조건 검사
        count=0
        for center in centers:
            x, y = center
            if y>0:
                count+=1
            if not (0 < x < 3 and ((1 < y < 3) or (-3 < y < -1))):
                return False

        if count != 1: # 같은 y영역에 있으면 안됨
            return False
            
        return True

def calculate_center_path(left_y_vals, right_y_vals, x_vals):
    """
    왼쪽과 오른쪽 경로의 y값을 평균내어 중앙 경로를 계산합니다.
    이 함수는 이제 원점을 지나지 않도록 조정된 y절편을 계산하지 않습니다.
    """
    center_y_vals = (left_y_vals + right_y_vals) / 2
    return center_y_vals

def fit_and_validate_path(data, x_vals, min_samples, residual_threshold, max_trials, path_type):
    """
    주어진 데이터에 대해 RANSAC 및 선형 회귀를 사용하여 경로를 추정.
    경로의 유형(왼쪽, 오른쪽, 중앙)에 따라 절편을 조정.
    """
    if len(data) >= max(2, int(0.2 * len(data))):
        fit_intercept = (path_type != 'center')  # Only set intercept to 0 for center path
        ransac = RANSACRegressor(
            LinearRegression(fit_intercept=fit_intercept),
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials
        )
        ransac.fit(data[:, 0].reshape(-1, 1), data[:, 1])
        
        if hasattr(ransac, 'estimator_'):  # Check if the estimator has been fitted
            validated_model = validate_and_adjust_model(ransac, x_vals, path_type)
            y_vals = validated_model.predict(x_vals)
            return y_vals
        else:
            print("RANSAC model fitting failed.")
            return None
    return None

def validate_and_adjust_model(ransac, x_vals, path_type):
    """
    RANSAC으로 얻은 모델의 기울기를 검증하고 조정.
    기울기가 설정된 범위를 벗어나면 조정하여 새로운 선형 회귀 모델 생성 및 반환.
    """
    if ransac.estimator_ is not None:
        m = ransac.estimator_.coef_[0]
        # 기울기 조정 범위 내에서 계산
        adjusted_m = max(min(m, 30), -30)
        
        # 조정된 기울기를 사용해 새로운 선형 회귀 모델 생성 및 훈련
        new_lr = LinearRegression()
        new_lr.coef_ = np.array([adjusted_m])
        new_lr.intercept_ = ransac.estimator_.intercept_ if path_type != 'center' else 0
        new_lr.fit(x_vals, ransac.predict(x_vals))  # x_vals는 입력 데이터

        return new_lr
    else:
        raise ValueError("Model has not been fitted yet.")

def publish_path(publisher, x_vals, y_vals, frame_id):
    path_msg = Path()
    path_msg.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    for x, y in zip(x_vals, y_vals):
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = x[0]
        pose.pose.position.y = y
        pose.pose.position.z = 0
        path_msg.poses.append(pose)
    publisher.publish(path_msg)
    #rospy.loginfo(f"Path published on {publisher.name}")

if __name__ == '__main__':
    try:
        rubber_cone = Rubber_cone()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node was shut down.")

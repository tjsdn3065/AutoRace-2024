#!/usr/bin/env python3

import rospy
import numpy as np
from morai_msgs.msg import CtrlCmd
from obstacle_detector.msg import Obstacles
from math import cos,sin,pi,sqrt,pow,atan2,tan
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped,Point

class Rubber_cone:
    def __init__(self):
        rospy.init_node('rubber_cone')
        rospy.Subscriber("/raw_obstacles", Obstacles, self.obstacle_callback)
        self.is_obstacles = False
        self.obstacles = []
        self.point_list = [] 

        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd', CtrlCmd, queue_size=1)

        self.center_path_pub = rospy.Publisher('/center_path', Path, queue_size=1)
        self.left_path_pub = rospy.Publisher('/left_path', Path, queue_size=1)
        self.right_path_pub = rospy.Publisher('/right_path', Path, queue_size=1)


        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2
        self.ctrl_cmd_msg.velocity = 5.0
        self.ctrl_cmd_msg.accel = 0.1
        self.ctrl_cmd_msg.steering = 0.0
        self.max_angle = 35*pi/180

        self.vehicle_length = 1.63
        self.lfd = 3

        self.waypoint_x = -1
        self.waypoint_y = -1

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.rubber_cone()
            rate.sleep()

    def obstacle_callback(self, msg):
        self.is_obstacles = True
        self.obstacles = msg.circles

        self.point_list = [] 

        for obstacle in self.obstacles:
            point=(obstacle.center.x,obstacle.center.y)
            self.point_list.append(point)

    def add_line_points(self, points, current_point, line, side):
        next_points = points[(points[:, 0] > current_point[0]) & (np.linalg.norm(points - current_point, axis=1) <= 2.0)]
        while next_points.size > 0:
            next_point = next_points[np.argmin(np.linalg.norm(next_points, axis=1))]
            if np.linalg.norm(next_point - current_point) <= 2.0:
                line.append(next_point)
                current_point = next_point
                next_points = points[(points[:, 0] > current_point[0]) & (np.linalg.norm(points - current_point, axis=1) <= 2.0)]
            else:
                break

    def get_index_based_center_point(self, left_line, right_line):
        left_size = len(left_line)
        right_size = len(right_line)

        if left_size == 0 and right_size == 0:
            return None, None
        elif left_size == 0:
            return None, right_line[-1]  # 오른쪽 라인의 마지막 포인트를 반환
        elif right_size == 0:
            return left_line[-1], None  # 왼쪽 라인의 마지막 포인트를 반환

        # 라인 간의 포인트 차이 계산
        difference = left_size - right_size
        n=abs(difference)
        if(n > 4):
            n = 4

        # 차이에 따른 중앙 포인트 설정
        if difference == 0:
            # 라인 길이가 같은 경우 각 라인의 첫 번째 포인트를 사용
            return left_line[0], right_line[0]
        elif difference > 0:
            # 왼쪽 라인이 더 긴 경우
            return left_line[n], right_line[0]
        else:
            # 오른쪽 라인이 더 긴 경우
            return left_line[0], right_line[n]

        # 위의 조건에 해당하지 않는 경우 (보안)
        return None, None


    def rubber_cone(self):
        if self.point_list:  # self.point_list가 비어있지 않은지 추가로 확인
            points = np.array(self.point_list)
            if points.size == 0:
                rospy.loginfo("No points received yet.")
                return
            
            # 왼쪽과 오른쪽 포인트 초기화 및 라인 구성
            left_line, right_line = [], []

            # 왼쪽 포인트 초기화
            left_points = points[(points[:, 1] > 0)] # y > 0
            if left_points.size > 0:
                current_point = left_points[np.argmin(np.linalg.norm(left_points, axis=1))] # 왼쪽에 있는 포인트 중에서 제일 가까운 포인트
                if np.linalg.norm(current_point) < 2.5: # 거리 2.5m 내
                    left_line.append(current_point)
                    self.add_line_points(points, current_point, left_line, 'left')

            # 오른쪽 포인트 초기화
            right_points = points[(points[:, 1] < 0)] # y < 0
            if right_points.size > 0:
                current_point = right_points[np.argmin(np.linalg.norm(right_points, axis=1))] # 오른쪽에 있는 포인트 중에서 제일 가까운 포인트
                if np.linalg.norm(current_point) < 2.5:
                    right_line.append(current_point)
                    self.add_line_points(points, current_point, right_line, 'right')

            # 경로 데이터 초기화
            right_path = Path()
            right_path.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
            if right_line != []:
                prev_x=0
                prev_y=0
                for point in right_line:
                    pose = PoseStamped()
                    pose.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                    pose.pose.position.x = point[0]
                    pose.pose.position.y = point[1]
                    pose.pose.position.z = 0
                    if prev_x != 0 and prev_y != 0:
                        between_pose = PoseStamped()
                        between_pose.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                        between_pose.pose.position.x = (point[0] + prev_x) /2
                        between_pose.pose.position.y = (point[1] + prev_y) /2
                        between_pose.pose.position.z = 0
                        right_path.poses.append(between_pose)
                    right_path.poses.append(pose)
                    prev_x=point[0]
                    prev_y=point[1]

            self.right_path_pub.publish(right_path)

            # 경로 데이터 초기화
            left_path = Path()
            left_path.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
            if left_line != []:
                prev_x=0
                prev_y=0
                for point in left_line:
                    pose = PoseStamped()
                    pose.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                    pose.pose.position.x = point[0]
                    pose.pose.position.y = point[1]
                    pose.pose.position.z = 0
                    if prev_x != 0 and prev_y != 0:
                        between_pose = PoseStamped()
                        between_pose.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                        between_pose.pose.position.x = (point[0] + prev_x) /2
                        between_pose.pose.position.y = (point[1] + prev_y) /2
                        between_pose.pose.position.z = 0
                        right_path.poses.append(between_pose)
                    left_path.poses.append(pose)
                    prev_x=point[0]
                    prev_y=point[1]

            self.left_path_pub.publish(left_path)

            # 경로 데이터 초기화
            center_line=[]
            if left_line != [] and right_line != []:
                left_size=len(left_line)
                right_size=len(right_line)

                if left_size > right_size:
                    # 더 작은 배열의 크기에 맞춰서 center_line을 계산
                    prev_x=0
                    prev_y=0
                    for i in range(right_size):
                        # 왼쪽과 오른쪽 라인의 평균 위치 계산
                        center_x = (left_line[i][0] + right_line[i][0]) / 2
                        center_y = (left_line[i][1] + right_line[i][1]) / 2
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

                    diff_x=center_line[right_size-1][0]-left_line[right_size-1][0]
                    diff_y=center_line[right_size-1][1]-left_line[right_size-1][1]

                    prev_x=0
                    prev_y=0
                    for i in range(right_size,left_size):
                        center_x = left_line[i][0] + diff_x
                        center_y = left_line[i][1] + diff_y
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

                elif left_size < right_size:
                    # 더 작은 배열의 크기에 맞춰서 center_line을 계산
                    prev_x=0
                    prev_y=0
                    for i in range(left_size):
                        # 왼쪽과 오른쪽 라인의 평균 위치 계산
                        center_x = (left_line[i][0] + right_line[i][0]) / 2
                        center_y = (left_line[i][1] + right_line[i][1]) / 2
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

                    diff_x=center_line[left_size-1][0]-right_line[left_size-1][0]
                    diff_y=center_line[left_size-1][1]-right_line[left_size-1][1]

                    prev_x=0
                    prev_y=0
                    for i in range(left_size,right_size):
                        center_x = right_line[i][0] + diff_x
                        center_y = right_line[i][1] + diff_y
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y
                else:
                    prev_x=0
                    prev_y=0
                    for left_point,right_point in zip(left_line,right_line):
                        center_x=(left_point[0]+right_point[0]) / 2
                        center_y=(left_point[1]+right_point[1]) / 2
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            center_line.append((between_center_x, between_center_y))
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

                # 경로 데이터 초기화
                center_path = Path()
                center_path.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                if center_line != []:
                    for point in center_line:
                        pose = PoseStamped()
                        pose.header = Header(stamp=rospy.Time.now(), frame_id='velodyne')
                        pose.pose.position.x = point[0]
                        pose.pose.position.y = point[1]
                        pose.pose.position.z = 0
                        center_path.poses.append(pose)

                self.center_path_pub.publish(center_path)

            # rospy.loginfo(f"Left line points: {left_line}")
            # rospy.loginfo(f"Right line points: {right_line}")

            # 각 라인의 포인트 개수 기반 중앙 포인트 추출
            left_center_point, right_center_point = self.get_index_based_center_point(left_line, right_line)

            if left_center_point is None and right_center_point is None:
                # rospy.logwarn("No valid points found, stopping")
                print("No valid points found, stopping")
                self.ctrl_cmd_msg.steering = 0
            else:
                if left_center_point is not None and right_center_point is None:
                    # rospy.loginfo("Only left point detected, turning right")
                    print("Only left point detected, turning right")
                    self.ctrl_cmd_msg.steering = -self.max_angle  # 오른쪽으로 큰 각도로 조향
                elif left_center_point is None and right_center_point is not None:
                    # rospy.loginfo("Only right point detected, turning left")
                    print("Only right point detected, turning left")
                    self.ctrl_cmd_msg.steering = self.max_angle  # 왼쪽으로 큰 각도로 조향
                else:
                    self.waypoint_x = (left_center_point[0] + right_center_point[0]) / 2.0
                    self.waypoint_y = (left_center_point[1] + right_center_point[1]) / 2.0

                    print(self.waypoint_x,self.waypoint_y)

                    theta = atan2(self.waypoint_y, self.waypoint_x)

                    if self.waypoint_x != 0:
                        self.ctrl_cmd_msg.steering = atan2(2 * self.vehicle_length * sin(theta), self.lfd)
                    else:
                        self.ctrl_cmd_msg.steering = 0 #if self.waypoint_y >= 0 else self.max_angle  # 방향에 따라 최대 각도 조정
                        print("?")
                    # rospy.loginfo("angle: %d", self.motor_msg.angle)
                    print(self.ctrl_cmd_msg.steering*180/pi)
                
            self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

if __name__ == '__main__':
    try:
        xycar = Rubber_cone()

    except rospy.ROSInternalException:
        pass 
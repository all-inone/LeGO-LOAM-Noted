#!/usr/bin/env python
import rospy
import numpy
from nav_msgs.msg import Path

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

ekfpath = Path()
gtpath = Path()
lidarpath =Path()

ekf_cb_flag = False
gt_cb_flag = False
lidar_cb_flag = False

def lidar_cb(data):
    print("hello")
    global lidar_cb_flag
    global lidarpath
    if lidar_cb_flag ==False:
        lidarpath =data
        print("hello")
        print(len(lidarpath.pose))
    lidar_cb_flag=True

def gt_cb(data):
    global gt_cb_flag
    global gtpath
    if gt_cb_flag ==False:
        gtpath=data
    gt_cb_flag=True  
      
def ekf_cb(data):
    global ekf_cb_flag
    global ekfpath
    if ekf_cb_flag==False:
        ekfpath=data
    ekf_cb_flag = True
 



if __name__ == '__main__':
    rospy.init_node('path_rovio')
    ekf_sub = rospy.Subscriber('/ekf_pose_path', Path, ekf_cb)
    gt_sub = rospy.Subscriber('/ground_truth_path', Path, gt_cb)
    lidar_sub = rospy.Subscriber('/lidar_path', Path, lidar_cb)

rospy.spin()
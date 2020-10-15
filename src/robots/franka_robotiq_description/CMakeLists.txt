cmake_minimum_required(VERSION 2.8.3)
project(franka_robotiq_description)

find_package(catkin REQUIRED
  actionlib_msgs
  actionlib
  controller_manager_msgs
  franka_control
  franka_description
  geometry_msgs
  panda_moveit_config
  robotiq_2f_gripper_control
  rosparam
  rospy
)
catkin_package(CATKIN_DEPENDS
  actionlib_msgs
  actionlib
  controller_manager_msgs
  franka_control
  franka_description
  geometry_msgs
  panda_moveit_config
  robotiq_2f_gripper_control
  rosparam
  rospy)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

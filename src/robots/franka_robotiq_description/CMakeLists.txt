cmake_minimum_required(VERSION 2.8.3)
project(franka_robotiq_description)

find_package(catkin REQUIRED)
catkin_package(CATKIN_DEPENDS xacro robotiq_2f_140_gripper_visualization)

install(DIRECTORY meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY robots
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

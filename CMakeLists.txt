cmake_minimum_required(VERSION 2.8.3)
project(jsklearn)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(DIRECTORY data img test
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS)

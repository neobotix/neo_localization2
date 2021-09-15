cmake_minimum_required(VERSION 3.5)
project(neo_localization2)

find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(neo_common2 REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(
  ${Boost_INCLUDE_DIRS}
  include
)

set(dependencies
  rclcpp
  geometry_msgs
  nav2_costmap_2d
  pluginlib
  nav_msgs
  nav2_util
  nav2_core
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  sensor_msgs
  nav_msgs
  neo_common2
)


set(library_name neo_localization_node)

add_executable(${library_name} src/neo_localization_node.cpp)

ament_target_dependencies(${library_name}
  ${dependencies}
)

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(${dependencies})
target_include_directories(${library_name}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/
   DESTINATION include/
)

install(TARGETS ${library_name}
DESTINATION lib/${PROJECT_NAME}
  )

install(DIRECTORY launch
    
DESTINATION share/${PROJECT_NAME})


ament_package()
cmake_minimum_required(VERSION 3.5)
project(neo_localization2)

find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_util REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(angles REQUIRED)

nav2_package()

set(executable_name neo_localization_node)
set(library_name ${executable_name}_core)

# Main library
add_library(${library_name} SHARED
  src/neo_localization_node.cpp
)

target_include_directories(${library_name}
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

target_link_libraries(${library_name} PUBLIC
  ${geometry_msgs_TARGETS}
  nav2_util::nav2_util_core
  angles::angles
  ${sensor_msgs_TARGETS}
  ${std_srvs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${nav_msgs_TARGETS}
  tf2_ros::tf2_ros
  tf2::tf2
  ${nav2_msgs_TARGETS}
)

target_link_libraries(${library_name} PRIVATE
  rclcpp_components::component
)

# Main executable
add_executable(${executable_name}
  src/main.cpp
)

target_link_libraries(${executable_name} PRIVATE ${library_name} rclcpp::rclcpp)

rclcpp_components_register_nodes(${library_name} "neo_localization2::NeoLocalizationNode")

install(
  TARGETS ${library_name} 
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${executable_name}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_libraries(${library_name})
ament_export_dependencies(
  geometry_msgs
  nav_msgs
  nav2_msgs
  nav2_util
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  angles
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()

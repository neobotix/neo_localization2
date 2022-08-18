cmake_minimum_required(VERSION 3.5)
project(neo_localization2)

find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(neo_common2 REQUIRED)
find_package(angles REQUIRED)


set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include_directories(
  ${Boost_INCLUDE_DIRS}
  include
)

set(dependencies
  rclcpp
  geometry_msgs
  nav_msgs
  tf2_ros
  angles
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
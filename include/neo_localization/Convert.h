/*
MIT License

Copyright (c) 2020 neobotix gmbh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#ifndef INCLUDE_NEO_LOCALIZATION_CONVERT_H_
#define INCLUDE_NEO_LOCALIZATION_CONVERT_H_

#include <neo_localization/GridMap.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> 
#include <tf2/transform_datatypes.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include "rclcpp/rclcpp.hpp"
#include <tf2/LinearMath/Transform.h>
#include <nav_msgs/msg/occupancy_grid.hpp>

/*
 * Converts ROS 3D Transform to a 2.5D matrix.
 */
inline
Matrix<double, 4, 4> convert_transform_25(const tf2::Transform& trans)
{
  Matrix<double, 4, 4> res;
  res(0, 0) = trans.getBasis()[0][0];
  res(1, 0) = trans.getBasis()[1][0];
  res(0, 1) = trans.getBasis()[0][1];
  res(1, 1) = trans.getBasis()[1][1];
  res(0, 3) = trans.getOrigin()[0];
  res(1, 3) = trans.getOrigin()[1];
  res(2, 3) = tf2::getYaw(trans.getRotation());
  res(2, 2) = 1;
  res(3, 3) = 1;
  return res;
}

/*
 * Converts ROS 3D Transform to a 3D matrix.
 */
inline
Matrix<double, 4, 4> convert_transform_3(const tf2::Transform& trans)
{
  Matrix<double, 4, 4> res;
  for(int j = 0; j < 3; ++j) {
    for(int i = 0; i < 3; ++i) {
      res(i, j) = trans.getBasis()[i][j];
    }
  }
  res(0, 3) = trans.getOrigin()[0];
  res(1, 3) = trans.getOrigin()[1];
  res(2, 3) = trans.getOrigin()[2];
  res(3, 3) = 1;
  return res;
}

/*
 * Converts a grid map to a ROS occupancy map.
 */
inline
std::shared_ptr<nav_msgs::msg::OccupancyGrid> convert_to_ros( std::shared_ptr<GridMap<float>> map,
                        Matrix<double, 3, 1> origin,
                        rclcpp::Time m_tCurrentTimeStamp)
{
  auto grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  tf2::Quaternion q;
  grid->header.stamp = m_tCurrentTimeStamp;
  grid->info.resolution = map->scale();
  grid->info.width = map->size_x();
  grid->info.height = map->size_y();
  grid->info.origin.position.x = origin[0];
  grid->info.origin.position.y = origin[1];
  q.setRPY(0, 0, origin[2]);
  grid->info.origin.orientation = tf2::toMsg(q);
  grid->data.resize(map->num_cells());
  for(int y = 0; y < map->size_y(); ++y) {
    for(int x = 0; x < map->size_x(); ++x) {
      grid->data[y * map->size_x() + x] = (*map)(x, y) * 100.f;
    }
  }
  return grid;
}

/*
 * Converts a grid map to a binary (-1, 0 or 100) ROS occupancy map.
 */
inline
std::shared_ptr<nav_msgs::msg::OccupancyGrid> convert_to_ros_binary(  std::shared_ptr<GridMap<float>> map,
                          Matrix<double, 3, 1> origin,
                          float threshold,
                          rclcpp::Time m_tCurrentTimeStamp)
{
  static const float coeff_55[5][5] = {
      {0.00118231, 0.01357, 0.0276652, 0.01357, 0.00118231},
      {0.01357, 0.06814, -0, 0.06814, 0.01357},
      {0.0276652, -0, -0.50349, -0, 0.0276652},
      {0.01357, 0.06814, -0, 0.06814, 0.01357},
      {0.00118231, 0.01357, 0.0276652, 0.01357, 0.00118231},
  };

  auto grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  grid->header.stamp = m_tCurrentTimeStamp;
  grid->info.resolution = map->scale();
  grid->info.width = map->size_x();
  grid->info.height = map->size_y();
  grid->info.origin.position.x = origin[0];
  grid->info.origin.position.y = origin[1];
  tf2::Quaternion q;
  q.setRPY(0, 0, origin[2]);
  grid->info.origin.orientation = tf2::toMsg(q);
  grid->data.resize(map->num_cells());
  for(int y = 0; y < map->size_y(); ++y) {
    for(int x = 0; x < map->size_x(); ++x) {
      // compute LoG filter
      float sum = 0;
      for(int j = -2; j <= 2; ++j) {
        const int y_ = std::min(std::max(y + j, 0), map->size_y() - 1);
        for(int i = -2; i <= 2; ++i) {
          const int x_ = std::min(std::max(x + i, 0), map->size_x() - 1);
          sum += coeff_55[j+2][i+2] * (*map)(x_, y_);
        }
      }
      if(-1 * sum > threshold) {
        grid->data[y * map->size_x() + x] = 100;
      } else if((*map)(x, y) < 0) {
        grid->data[y * map->size_x() + x] = -1;
      } else {
        grid->data[y * map->size_x() + x] = 0;
      }
    }
  }
  return grid;
}


#endif /* INCLUDE_NEO_LOCALIZATION_CONVERT_H_ */

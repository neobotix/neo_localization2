/*
 * neo_localization_node.cpp
 *
 *  Created on: Apr 8, 2020
 *      Author: mad
 */
#include <neo_localization/Util.h>
#include <neo_localization/Convert.h>
#include <neo_localization/Solver.h>
#include <neo_localization/GridMap.h>

#include "rclcpp/rclcpp.hpp"
#include <rclcpp/node_options.hpp>

#include <angles/angles.h>
#include <nav_msgs/msg/odometry.h>
#include <nav_msgs/msg/occupancy_grid.h>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/quaternion.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.h>
#include <geometry_msgs/msg/pose_with_covariance_stamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <chrono>
#include <memory>

using namespace std::chrono_literals;

#include <mutex>
#include <thread>
#include <random>
#include <cmath>
#include <array>
#include <tf2_ros/create_timer_ros.h>
using std::placeholders::_1;
using std::placeholders::_2;

/*
 * Coordinate systems:
 * - Sensor in [meters, rad], aka. "laserX"
 * - Base Link in [meters, rad], aka. "base_link"
 * - Odometry in [meters, rad], aka. "odom"
 * - Map in [meters, rad], aka. "map"
 * - World Grid in [meters, rad], aka. "world"
 * - Tile Grid in [meters, rad], aka. "grid"
 * - World Grid in [pixels]
 * - Tile Grid in [pixels]
 *
 */
class NeoLocalizationNode : public rclcpp::Node {
public:
  NeoLocalizationNode(): Node("neo_localization_node")
  {
    this->declare_parameter<bool>("broadcast_tf", true);
    this->get_parameter_or("broadcast_tf", m_broadcast_tf, true);

    this->declare_parameter<std::string>("base_frame", "base_link");
    this->get_parameter("base_frame", m_base_frame);

    this->declare_parameter<std::string>("odom_frame", "odom");
    this->get_parameter("odom_frame", m_odom_frame);

    this->declare_parameter<std::string>("map_frame", "map");
    this->get_parameter("map_frame", m_map_frame);

    this->declare_parameter<std::string>("map_topic", "map");
    this->get_parameter("map_topic", m_map_topic);

    this->declare_parameter<int>("map_size", 1000);
    this->get_parameter("map_size", m_map_size);

    this->declare_parameter<int>("map_downscale", 0);
    this->get_parameter("map_downscale", m_map_downscale);

    this->declare_parameter<int>("num_smooth", 5);
    this->get_parameter("num_smooth", m_num_smooth);

    this->declare_parameter<int>("sample_rate", 5);
    this->get_parameter("sample_rate", m_sample_rate);

    this->declare_parameter<int>("solver_iterations", 5);
    this->get_parameter("solver_iterations", m_solver_iterations);

    this->declare_parameter<int>("min_points", 5);
    this->get_parameter("min_points", m_min_points);

    this->declare_parameter<double>("map_update_rate", 0.5);
    this->get_parameter("map_update_rate", m_map_update_rate);

    this->declare_parameter<int>("loc_update_time", 100.);
    this->get_parameter("loc_update_time", m_loc_update_time_ms);

    this->declare_parameter<double>("min_score", 0.2);
    this->get_parameter("min_score", m_min_score);

    this->declare_parameter<double>("solver_gain", 0.1);
    this->get_parameter("solver_gain", m_solver.gain);

    this->declare_parameter<double>("solver_damping", 1000);
    this->get_parameter("solver_damping", m_solver.damping);

    this->declare_parameter<double>("update_gain", 0.5);
    this->get_parameter("update_gain", m_update_gain);

    this->declare_parameter<double>("confidence_gain", 0.01);
    this->get_parameter("confidence_gain", m_confidence_gain);

    this->declare_parameter<double>("odometry_std_xy", 0.01);
    this->get_parameter("odometry_std_xy", m_odometry_std_xy);

    this->declare_parameter<double>("odometry_std_yaw", 0.01);
    this->get_parameter("odometry_std_yaw", m_odometry_std_yaw);

    this->declare_parameter<double>("min_sample_std_xy", 0.025);
    this->get_parameter("min_sample_std_xy", m_min_sample_std_xy);

    this->declare_parameter<double>("min_sample_std_yaw", 0.025);
    this->get_parameter("min_sample_std_yaw", m_min_sample_std_yaw);

    this->declare_parameter<double>("max_sample_std_xy", 0.5);
    this->get_parameter("max_sample_std_xy", m_max_sample_std_xy);

    this->declare_parameter<double>("max_sample_std_yaw", 0.5);
    this->get_parameter("max_sample_std_yaw", m_max_sample_std_yaw);

    this->declare_parameter<double>("constrain_threshold", 0.1);
    this->get_parameter("constrain_threshold", m_constrain_threshold);

    this->declare_parameter<double>("constrain_threshold_yaw", 0.2);
    this->get_parameter("constrain_threshold_yaw", m_constrain_threshold_yaw);

    this->declare_parameter<double>("transform_timeout", 0.2);
    this->get_parameter("transform_timeout", m_transform_timeout);

    this->declare_parameter<std::string>("scan_topic", "scan");
    this->get_parameter("scan_topic", m_scan_topic);

    this->declare_parameter<std::string>("initialpose", "initialpose");
    this->get_parameter("initialpose", m_initial_pose);

    this->declare_parameter<std::string>("map_tile", "map_tile");
    this->get_parameter("map_tile", m_map_tile);

    this->declare_parameter<std::string>("map_pose", "map_pose");
    this->get_parameter("map_pose", m_map_pose);

    this->declare_parameter<std::string>("particle_cloud", "particlecloud");
    this->get_parameter("particle_cloud", m_particle_cloud);

    this->declare_parameter<std::string>("amcl_pose", "amcl_pose");
    this->get_parameter("amcl_pose", m_amcl_pose);

    this->declare_parameter<bool>("broadcast_info", false);
    this->get_parameter("broadcast_info", m_broadcast_info);

    m_map_update_thread = std::thread(&NeoLocalizationNode::update_loop, this);

    m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    m_sub_scan_topic = this->create_subscription<sensor_msgs::msg::LaserScan>(m_scan_topic, rclcpp::SensorDataQoS(), std::bind(&NeoLocalizationNode::scan_callback, this, _1));
    m_sub_map_topic = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(), std::bind(&NeoLocalizationNode::map_callback, this, _1));
    m_sub_pose_estimate = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(m_initial_pose, 1, std::bind(&NeoLocalizationNode::pose_callback, this, _1));

    m_pub_map_tile = this->create_publisher<nav_msgs::msg::OccupancyGrid>(m_map_tile, 1);
    m_pub_loc_pose = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(m_amcl_pose, 10);
    m_pub_loc_pose_2 = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(m_map_pose, 10);
    m_pub_pose_array = this->create_publisher<geometry_msgs::msg::PoseArray>(m_particle_cloud, 10);

    m_loc_update_timer = create_wall_timer(
                std::chrono::milliseconds(m_loc_update_time_ms), std::bind(&NeoLocalizationNode::loc_update, this));


    buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());

    transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer);

    std::string robot_namespace(this->get_namespace());

    // removing the unnecessary "/" from the namespace
    robot_namespace.erase(std::remove(robot_namespace.begin(), robot_namespace.end(), '/'), 
    robot_namespace.end());

    m_base_frame = robot_namespace + m_base_frame;
    m_odom_frame = robot_namespace + m_odom_frame;
  }

  ~NeoLocalizationNode()
  {
    if(m_map_update_thread.joinable()) {
    }
  }

protected:

  /*
   * Computes localization update for a single laser scan.
   */
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
  {

    std::lock_guard<std::mutex> lock(m_node_mutex);

    if(!map_received_) {
      RCLCPP_INFO_ONCE(this->get_logger(), "no map");
      return;
    }
    RCLCPP_INFO_ONCE(this->get_logger(), "map_received");
    scan->header.frame_id = m_ns + scan->header.frame_id;
    m_scan_buffer[scan->header.frame_id] = scan;
  }

  /*
   * Convert/Transform a scan from ROS format to a specified base frame.
   */
  std::vector<scan_point_t> convert_scan(const sensor_msgs::msg::LaserScan::SharedPtr scan, const Matrix<double, 4, 4>& odom_to_base)
  {
    std::vector<scan_point_t> points;
    tf2::Stamped<tf2::Transform> base_to_odom;
    tf2::Stamped<tf2::Transform> sensor_to_base;
    bool callback_timeout = false;
    try {
      auto tempTransform = buffer->lookupTransform(m_base_frame, scan->header.frame_id, tf2::TimePointZero);
      tf2::fromMsg(tempTransform, sensor_to_base);

    } catch(const std::exception& ex) {
      RCLCPP_WARN_STREAM(this->get_logger(), "NeoLocalizationNode: lookupTransform(scan->header.frame_id, m_base_frame) failed: " << ex.what());
      return points;
    }
    try {
      auto tempTransform = buffer->lookupTransform(m_odom_frame, m_base_frame, scan->header.stamp, tf2::durationFromSec(m_transform_timeout));
      tf2::fromMsg(tempTransform, base_to_odom);
      } catch(const std::exception& ex) {
      RCLCPP_WARN_STREAM(this->get_logger(), "NeoLocalizationNode: lookupTransform(m_base_frame, m_odom_frame) failed: " << ex.what());
      return points;
    }
    
    const Matrix<double, 4, 4> S = convert_transform_3(sensor_to_base);
    const Matrix<double, 4, 4> L = convert_transform_25(base_to_odom);

    // precompute transformation matrix from sensor to requested base
    const Matrix<double, 4, 4> T = odom_to_base * L * S;

    for(size_t i = 0; i < scan->ranges.size(); ++i)
    {
      if(scan->ranges[i] <= scan->range_min || scan->ranges[i] >= scan->range_max) {
        continue; // no actual measurement
      }

      // transform sensor points into base coordinate system
      const Matrix<double, 3, 1> scan_pos = (T * rotate3_z<double>(scan->angle_min + i * scan->angle_increment)
                          * Matrix<double, 4, 1>{scan->ranges[i], 0, 0, 1}).project();
      scan_point_t point;
      point.x = scan_pos[0];
      point.y = scan_pos[1];
      points.emplace_back(point);
    }
    return points;
  }

  void loc_update()
  {
    std::lock_guard<std::mutex> lock(m_node_mutex);
    if(!map_received_ || m_scan_buffer.empty() || !m_initialized) {
      return;
    }

    tf2::Stamped<tf2::Transform> base_to_odom;

    try {
      auto tempTransform = buffer->lookupTransform(m_odom_frame, m_base_frame, tf2::TimePointZero);
      tf2::fromMsg(tempTransform, base_to_odom);
    } catch(const std::exception& ex) {
      RCLCPP_WARN_STREAM(this->get_logger(), "NeoLocalizationNode: lookup Transform(m_base_frame, m_odom_frame) failed: " << ex.what());
      return;
    }
    
    tf2::Transform base_to_odom_ws(base_to_odom.getRotation(), base_to_odom.getOrigin());
    
    const Matrix<double, 4, 4> L = convert_transform_25(base_to_odom_ws);
    const Matrix<double, 4, 4> T = translate25(m_offset_x, m_offset_y) * rotate25_z(m_offset_yaw);    // odom to map

    const Matrix<double, 3, 1> odom_pose = (L * Matrix<double, 4, 1>{0, 0, 0, 1}).project();
    const double dist_moved = (odom_pose - m_last_odom_pose).get<2>().norm();
    const double rad_rotated = fabs(angles::normalize_angle(odom_pose[2] - m_last_odom_pose[2]));

    std::vector<scan_point_t> points;

    RCLCPP_INFO_ONCE(this->get_logger(), "map_received");
    // convert all scans to current base frame
    for(const auto& scan : m_scan_buffer)
    {
      
      auto scan_points = convert_scan(scan.second, L.inverse());

      points.insert(points.end(), scan_points.begin(), scan_points.end());
    }

    // // check for number of points
    if(points.size() < m_min_points)
    {
      RCLCPP_WARN_STREAM(this->get_logger(),"NeoLocalizationNode: Number of points too low: " << points.size());
      return;
    }

    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp = tf2_ros::toMsg(base_to_odom.stamp_);
    pose_array.header.frame_id = m_map_frame;

    // calc predicted grid pose based on odometry

    const Matrix<double, 3, 1> grid_pose = (m_grid_to_map.inverse() * T * L * Matrix<double, 4, 1>{0, 0, 0, 1}).project();
    // setup distributions
    std::normal_distribution<double> dist_x(grid_pose[0], m_sample_std_xy);
    std::normal_distribution<double> dist_y(grid_pose[1], m_sample_std_xy);
    std::normal_distribution<double> dist_yaw(grid_pose[2], m_sample_std_yaw);

    // solve odometry prediction first
    m_solver.pose_x = grid_pose[0];
    m_solver.pose_y = grid_pose[1];
    m_solver.pose_yaw = grid_pose[2];

    for(int iter = 0; iter < m_solver_iterations; ++iter) {
      m_solver.solve<float>(*m_map, points);
    }

    double best_x = m_solver.pose_x;
    double best_y = m_solver.pose_y;
    double best_yaw = m_solver.pose_yaw;
    double best_score = m_solver.r_norm;

    std::vector<Matrix<double, 3, 1>> seeds(m_sample_rate);
    std::vector<Matrix<double, 3, 1>> samples(m_sample_rate);
    std::vector<double> sample_errors(m_sample_rate);

    for(int i = 0; i < m_sample_rate; ++i)
    {
      // generate new sample
      m_solver.pose_x = dist_x(m_generator);
      m_solver.pose_y = dist_y(m_generator);
      m_solver.pose_yaw = dist_yaw(m_generator);

      seeds[i] = Matrix<double, 3, 1>{m_solver.pose_x, m_solver.pose_y, m_solver.pose_yaw};

      // solve sample
      for(int iter = 0; iter < m_solver_iterations; ++iter) {
        m_solver.solve<float>(*m_map, points);
      }

      // save sample
      const auto sample = Matrix<double, 3, 1>{m_solver.pose_x, m_solver.pose_y, m_solver.pose_yaw};
      samples[i] = sample;
      sample_errors[i] = m_solver.r_norm;

      // check if sample is better
      if(m_solver.r_norm > best_score) {
        best_x = m_solver.pose_x;
        best_y = m_solver.pose_y;
        best_yaw = m_solver.pose_yaw;
        best_score = m_solver.r_norm;
      }

      // add to visualization
      {
        const Matrix<double, 3, 1> map_pose = (m_grid_to_map * sample.extend()).project();
        tf2::Quaternion tmp;
        geometry_msgs::msg::Pose pose;
        pose.position.x = map_pose[0];
        pose.position.y = map_pose[1];
        tmp.setRPY( 0, 0, map_pose[2]);
        auto tmp_msg = tf2::toMsg(tmp);
        pose.orientation = tmp_msg;
        pose_array.poses.push_back(pose);
      }
    }

    // compute covariances
    double mean_score = 0;
    Matrix<double, 3, 1> mean_xyw;
    Matrix<double, 3, 1> seed_mean_xyw;
    const double var_error = compute_variance(sample_errors, mean_score);
    const Matrix<double, 3, 3> var_xyw = compute_covariance(samples, mean_xyw);
    const Matrix<double, 3, 3> grad_var_xyw =
        compute_virtual_scan_covariance_xyw(m_map, points, Matrix<double, 3, 1>{best_x, best_y, best_yaw});

    // compute gradient characteristic
    std::array<Matrix<double, 2, 1>, 2> grad_eigen_vectors;
    const Matrix<double, 2, 1> grad_eigen_values = compute_eigenvectors_2(grad_var_xyw.get<2, 2>(), grad_eigen_vectors);
    const Matrix<double, 3, 1> grad_std_uvw{sqrt(grad_eigen_values[0]), sqrt(grad_eigen_values[1]), sqrt(grad_var_xyw(2, 2))};

    // decide if we have 3D, 2D, 1D or 0D localization
    int mode = 0;
    if(best_score > m_min_score) {
      if(grad_std_uvw[0] > m_constrain_threshold) {
        if(grad_std_uvw[1] > m_constrain_threshold) {
          mode = 3; // 2D position + rotation
        } else if(grad_std_uvw[2] > m_constrain_threshold_yaw) {
          mode = 2; // 1D position + rotation
        } else {
          mode = 1; // 1D position only
        }
      }
    }

    if(mode > 0)
    {
      double new_grid_x = best_x;
      double new_grid_y = best_y;
      double new_grid_yaw = best_yaw;

      if(mode < 3)
      {
        // constrain update to the good direction (ie. in direction of the eigen vector with the smaller sigma)
        const auto delta = Matrix<double, 2, 1>{best_x, best_y} - Matrix<double, 2, 1>{grid_pose[0], grid_pose[1]};
        const auto dist = grad_eigen_vectors[0].dot(delta);
        new_grid_x = grid_pose[0] + dist * grad_eigen_vectors[0][0];
        new_grid_y = grid_pose[1] + dist * grad_eigen_vectors[0][1];
      }
      if(mode < 2) {
        new_grid_yaw = grid_pose[2];  // keep old orientation
      }

      // use best sample for update
      Matrix<double, 4, 4> grid_pose_new = translate25(new_grid_x, new_grid_y) * rotate25_z(new_grid_yaw);

      // compute new odom to map offset from new grid pose
      const Matrix<double, 3, 1> new_offset =
          (m_grid_to_map * grid_pose_new * L.inverse() * Matrix<double, 4, 1>{0, 0, 0, 1}).project();

      // apply new offset with an exponential low pass filter
      m_offset_x += (new_offset[0] - m_offset_x) * m_update_gain;
      m_offset_y += (new_offset[1] - m_offset_y) * m_update_gain;
      m_offset_yaw += angles::shortest_angular_distance(m_offset_yaw, new_offset[2]) * m_update_gain;
    }
    m_offset_time = tf2_ros::toMsg(base_to_odom.stamp_);

    // update particle spread depending on mode
    if(mode >= 3) {
      m_sample_std_xy *= (1 - m_confidence_gain);
    } else {
      m_sample_std_xy += dist_moved * m_odometry_std_xy;
    }
    if(mode >= 2) {
      m_sample_std_yaw *= (1 - m_confidence_gain);
    } else {
      m_sample_std_yaw += rad_rotated * m_odometry_std_yaw;
    }

    // limit particle spread
    m_sample_std_xy = fmin(fmax(m_sample_std_xy, m_min_sample_std_xy), m_max_sample_std_xy);
    m_sample_std_yaw = fmin(fmax(m_sample_std_yaw, m_min_sample_std_yaw), m_max_sample_std_yaw);

    // publish new transform
    broadcast();

    const Matrix<double, 3, 1> new_map_pose = (translate25(m_offset_x, m_offset_y) * rotate25_z(m_offset_yaw) *
                          L * Matrix<double, 4, 1>{0, 0, 0, 1}).project();
    tf2::Quaternion myQuaternion;
    // publish localization pose
    geometry_msgs::msg::PoseWithCovarianceStamped loc_pose;
    loc_pose.header.stamp = m_offset_time;
    loc_pose.header.frame_id = m_map_frame;
    loc_pose.pose.pose.position.x = new_map_pose[0];
    loc_pose.pose.pose.position.y = new_map_pose[1];
    loc_pose.pose.pose.position.z = 0;
    myQuaternion.setRPY(0, 0, new_map_pose[2]);
    auto temp_quat = tf2::toMsg(myQuaternion);
    loc_pose.pose.pose.orientation = temp_quat;
    for(int j = 0; j < 3; ++j) {
      for(int i = 0; i < 3; ++i) {
        const int i_ = (i == 2 ? 5 : i);
        const int j_ = (j == 2 ? 5 : j);
        loc_pose.pose.covariance[j_ * 6 + i_] = var_xyw(i, j);
      }
    }
    m_pub_loc_pose->publish(loc_pose);
    m_pub_loc_pose_2->publish(loc_pose);

    // publish visualization
    m_pub_pose_array->publish(pose_array);

    // keep last odom pose
    m_last_odom_pose = odom_pose;

    if(m_broadcast_info == true) {
      if(update_counter++ % 10 == 0) {
        RCLCPP_INFO_STREAM(this->get_logger(),  "NeoLocalizationNode: score=" << float(best_score) << ", grad_uvw=[" << float(grad_std_uvw[0]) << ", " << float(grad_std_uvw[1])
          << ", " << float(grad_std_uvw[2]) << "], std_xy=" << float(m_sample_std_xy) << " m, std_yaw=" << float(m_sample_std_yaw)
          << " rad, mode=" << mode << "D, " << m_scan_buffer.size() << " scans");
      }
    }

    // clear scan buffer
    m_scan_buffer.clear();
  }

  /*
   * Resets localization to given position.
   */
  void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose)
  {
    {
      std::lock_guard<std::mutex> lock(m_node_mutex);

      if(pose->header.frame_id != m_map_frame) {
        RCLCPP_WARN_STREAM(this->get_logger(), "NeoLocalizationNode: Invalid pose estimate frame");
        return;
      }

      tf2::Stamped<tf2::Transform> base_to_odom;
      tf2::Transform map_pose;
      tf2::fromMsg(pose->pose.pose, map_pose);

      RCLCPP_INFO_STREAM(this->get_logger(), "NeoLocalizationNode: Got new map pose estimate: x=" << map_pose.getOrigin()[0]
              << " m, y=" <<  map_pose.getOrigin()[1] );

      try {
        auto tempTransform = buffer->lookupTransform(m_odom_frame, m_base_frame, tf2::TimePointZero);
        tf2::convert(tempTransform, base_to_odom);
      } catch(const std::exception& ex) {
        RCLCPP_WARN_STREAM(this->get_logger(),"NeoLocalizationNode: lookupTransform(m_base_frame, m_odom_frame) failed: "<< ex.what());
        return;
      }

      const Matrix<double, 4, 4> L = convert_transform_25(base_to_odom);

      // compute new odom to map offset
      const Matrix<double, 3, 1> new_offset =
          (convert_transform_25(map_pose) * L.inverse() * Matrix<double, 4, 1>{0, 0, 0, 1}).project();

      // set new offset based on given position
      m_offset_x = new_offset[0];
      m_offset_y = new_offset[1];
      m_offset_yaw = new_offset[2];

      // reset particle spread to maximum
      m_sample_std_xy = m_max_sample_std_xy;
      m_sample_std_yaw = m_max_sample_std_yaw;

      broadcast();
    }

    // get a new map tile immediately
    update_map();
  }

  /*
   * Stores the given map.
   */
  void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr ros_map)
  {
    std::lock_guard<std::mutex> lock(m_node_mutex);
    map_received_ = true;

    RCLCPP_INFO_STREAM(this->get_logger(), "NeoLocalizationNode: Got new map with dimensions " << ros_map->info.width << " x " << ros_map->info.height
        << " and cell size " << ros_map->info.resolution);

    {
      tf2::Transform tmp;
      tf2::fromMsg(ros_map->info.origin, tmp);
      m_world_to_map = convert_transform_25(tmp);
    }
    m_world = ros_map;
    // reset particle spread to maximum
    m_sample_std_xy = m_max_sample_std_xy;
    m_sample_std_yaw = m_max_sample_std_yaw;
  }

  /*
   * Extracts a new map tile around current position.
   */
  void update_map()
  {
    Matrix<double, 4, 4> world_to_map;      // transformation from original grid map (integer coords) to "map frame"
    Matrix<double, 3, 1> world_pose;      // pose in the original (integer coords) grid map (not map tile)
    nav_msgs::msg::OccupancyGrid::SharedPtr world;
    {
      std::lock_guard<std::mutex> lock(m_node_mutex);
      if(!m_world) {
        return;
      }

      tf2::Stamped<tf2::Transform> base_to_odom;
      try {
        auto tempTransform = buffer->lookupTransform(m_odom_frame, m_base_frame, tf2::TimePointZero);
        tf2::fromMsg(tempTransform, base_to_odom);
      } catch(const std::exception& ex) {
        RCLCPP_WARN_STREAM(this->get_logger(),"NeoLocalizationNode: lookupTransform(m_base_frame, m_odom_frame) failed: " << ex.what());
        return;
      }

      const Matrix<double, 4, 4> L = convert_transform_25(base_to_odom);
      const Matrix<double, 4, 4> T = translate25(m_offset_x, m_offset_y) * rotate25_z(m_offset_yaw);    // odom to map
      world_pose = (m_world_to_map.inverse() * T * L * Matrix<double, 4, 1>{0, 0, 0, 1}).project();

      world = m_world;
      world_to_map = m_world_to_map;
    }

    // compute tile origin in pixel coords
    const double world_scale = world->info.resolution;
    const int tile_x = int(world_pose[0] / world_scale) - m_map_size / 2;
    const int tile_y = int(world_pose[1] / world_scale) - m_map_size / 2;

    auto map = std::make_shared<GridMap<float>>(m_map_size, m_map_size, world_scale);

    // extract tile and convert to our format (occupancy between 0 and 1)
    for(int y = 0; y < map->size_y(); ++y) {
      for(int x = 0; x < map->size_x(); ++x) {
        const int x_ = std::min(std::max(tile_x + x, 0), int(world->info.width) - 1);
        const int y_ = std::min(std::max(tile_y + y, 0), int(world->info.height) - 1);
        const auto cell = world->data[y_ * world->info.width + x_];
        if(cell >= 0) {
          (*map)(x, y) = fminf(cell / 100.f, 1.f);
        } else {
          (*map)(x, y) = 0;
        }
      }
    }

    // optionally downscale map
    for(int i = 0; i < m_map_downscale; ++i) {
      map = map->downscale();
    }

    // smooth map
    for(int i = 0; i < m_num_smooth; ++i) {
      map->smooth_33_1();
    }

    // update map
    {
      std::lock_guard<std::mutex> lock(m_node_mutex);
      m_map = map;
      m_grid_to_map = world_to_map * translate25<double>(tile_x * world_scale, tile_y * world_scale);
      m_initialized = true;
    }

    const auto tile_origin = (m_grid_to_map * Matrix<double, 4, 1>{0, 0, 0, 1}).project();
    const auto tile_center = (m_grid_to_map * Matrix<double, 4, 1>{ map->scale() * map->size_x() / 2,
                                    map->scale() * map->size_y() / 2, 0, 1}).project();

    // publish new map tile for visualization
    tf2::Quaternion myQuaternion;
    nav_msgs::msg::OccupancyGrid ros_grid;
    ros_grid.header.stamp = m_offset_time;
    ros_grid.header.frame_id = m_map_frame;
    ros_grid.info.resolution = map->scale();
    ros_grid.info.width = map->size_x();
    ros_grid.info.height = map->size_y();
    ros_grid.info.origin.position.x = tile_origin[0];
    ros_grid.info.origin.position.y = tile_origin[1];
    tf2::Quaternion Quaternion1;
    myQuaternion.setRPY( 0, 0, tile_origin[2]);
    ros_grid.info.origin.orientation = tf2::toMsg(myQuaternion);
    ros_grid.data.resize(map->num_cells());
    for(int y = 0; y < map->size_y(); ++y) {
      for(int x = 0; x < map->size_x(); ++x) {
        ros_grid.data[y * map->size_x() + x] = (*map)(x, y) * 100.f;
      }
    }
    m_pub_map_tile->publish(ros_grid);

  }

  /*
   * Asynchronous map update loop, running in separate thread.
   */
  void update_loop()
  {
    RCLCPP_INFO_ONCE(this->get_logger(),"NeoLocalizationNode: Activating map update loop");

    rclcpp::Rate loop_rate(m_map_update_rate);
    while(rclcpp::ok()) {
      try {
        update_map(); // get a new map tile periodically
      }
      catch(const std::exception& ex) {
        RCLCPP_WARN_STREAM(this->get_logger(),"NeoLocalizationNode: update_map() failed:");
      }
      loop_rate.sleep();
    }
  }

  /*
   * Publishes "map" frame on tf_->
   */
  void broadcast()
  {
    if(m_broadcast_tf)
    {
      // compose and publish transform for tf package
      geometry_msgs::msg::TransformStamped pose;
      // compose header
      // Adding an expiry time of the frame. Same procedure followed in nav2_amcl
      m_offset_time.nanosec = m_offset_time.nanosec + 1000000000;
      pose.header.stamp = m_offset_time;
      pose.header.frame_id = m_map_frame;
      pose.child_frame_id = m_odom_frame;
      // compose data container
      pose.transform.translation.x = m_offset_x;
      pose.transform.translation.y = m_offset_y;
      pose.transform.translation.z = 0;
      tf2::Quaternion myQuaternion;
      myQuaternion.setRPY( 0, 0, m_offset_yaw);
      pose.transform.rotation = tf2::toMsg(myQuaternion);

      // publish the transform
      m_tf_broadcaster->sendTransform(pose);
    }
  }

private:
  std::mutex m_node_mutex;

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr m_pub_map_tile;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_pub_loc_pose;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_pub_loc_pose_2;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr m_pub_pose_array;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr m_sub_map_topic;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_sub_scan_topic;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_sub_pose_estimate;
  std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

  bool m_broadcast_tf = false;
  bool m_initialized = false;
  std::string m_base_frame;
  std::string m_odom_frame;
  std::string m_map_frame;
  std::string m_map_topic;
  std::string m_scan_topic;
  std::string m_initial_pose;
  std::string m_map_tile;
  std::string m_map_pose;
  std::string m_particle_cloud;
  std::string m_amcl_pose;
  std::string m_ns = "";

  int m_map_size = 0;
  int m_map_downscale = 0;
  int m_num_smooth = 0;
  int m_solver_iterations = 0;
  int m_sample_rate = 0;
  int m_min_points = 0;
  double m_update_gain = 0;
  double m_confidence_gain = 0;
  double m_min_score = 0;
  double m_odometry_std_xy = 0;     // odometry xy error in meter per meter driven
  double m_odometry_std_yaw = 0;      // odometry yaw error in rad per rad rotated
  double m_min_sample_std_xy = 0;
  double m_min_sample_std_yaw = 0;
  double m_max_sample_std_xy = 0;
  double m_max_sample_std_yaw = 0;
  double m_constrain_threshold = 0;
  double m_constrain_threshold_yaw = 0;
  int m_loc_update_time_ms = 0;
  double m_map_update_rate = 0;
  double m_transform_timeout = 0;

  builtin_interfaces::msg::Time m_offset_time;
  double m_offset_x = 0;          // current x offset between odom and map
  double m_offset_y = 0;          // current y offset between odom and map
  double m_offset_yaw = 0;        // current yaw offset between odom and map
  double m_sample_std_xy = 0;       // current sample spread in xy
  double m_sample_std_yaw = 0;      // current sample spread in yaw
  std::unique_ptr<tf2_ros::Buffer> buffer;
  std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};

  Matrix<double, 3, 1> m_last_odom_pose;
  Matrix<double, 4, 4> m_grid_to_map;
  Matrix<double, 4, 4> m_world_to_map;
  std::shared_ptr<GridMap<float>> m_map;      // map tile
  nav_msgs::msg::OccupancyGrid::SharedPtr m_world;    // whole map
  bool map_received_ = false;

  int64_t update_counter = 0;
  std::map<std::string, sensor_msgs::msg::LaserScan::SharedPtr> m_scan_buffer;

  Solver m_solver;
  std::mt19937 m_generator;
  std::thread m_map_update_thread;
  bool m_broadcast_info;
  rclcpp::TimerBase::SharedPtr m_loc_update_timer;

};

int main(int argc, char** argv)
{
  // initialize ROS
  rclcpp::init(argc, argv);
  
  try {
    auto nh = std::make_shared<NeoLocalizationNode>() ;
    rclcpp::spin(nh);
  }
  catch(const std::exception& ex) {
    std::cout<<"NeoLocalizationNode: " << ex.what() << std::endl;
    return -1;
  }

  return 0;
}

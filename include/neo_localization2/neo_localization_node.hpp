#ifndef NEO_LOCALIZATION2__NEO_LOCALIZATION_NODE_HPP_
#define NEO_LOCALIZATION2__NEO_LOCALIZATION_NODE_HPP_

#include "neo_localization2/utils/Util.hpp"
#include "neo_localization2/utils/Convert.hpp"
#include "neo_localization2/solver/Solver.hpp"
#include "neo_localization2/map/GridMap.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/node_options.hpp"
#include "angles/angles.h"
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2_ros/buffer.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/quaternion.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <random>
#include <cmath>
#include <array>
#include <tf2_ros/create_timer_ros.hpp>

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

namespace neo_localization2
{
/*
 * @class NeoLocalizationNode
 * @brief ROS wrapper for NeoLocalization
 */
class NeoLocalizationNode: public nav2_util::LifecycleNode
{
public:

  /*
   * @brief NeoLocalization constructor
   * @param options Additional options to control creation of the node.
   */
  explicit NeoLocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  /*
   * @brief NeoLocalization destructor
   */
  ~NeoLocalizationNode();

protected:
  /*
   * @brief Lifecycle configure
   */
  nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  /*
   * @brief Lifecycle activate
   */
  nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  /*
   * @brief Lifecycle deactivate
   */
  nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  /*
   * @brief Lifecycle cleanup
   */
  nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  /*
   * @brief Lifecycle shutdown
   */
  nav2_util::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

  /*
   * @brief Initialize parameters
   */
  void initParameters(nav2_util::LifecycleNode::SharedPtr node);

  /*
   * @brief Get parameters
   */
  void getParameters(nav2_util::LifecycleNode::SharedPtr node);

  /*
   * @brief Initialize transforms
   */
  void initTransforms();

  /*
   * @brief Initialize publishers & subscribers
   */
  void initPubSub();

  /*
   * @brief Initialize robot namespace
   */
  void initNameSpace();

  /*
   * Computes localization update for a single laser scan.
   */
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan);

  /*
   * Whether just only use odom (without scan matching) or not in localization
   */
  void use_odom_callback(const std_msgs::msg::Bool::SharedPtr msg);

  /*
   * Convert/Transform a scan from ROS format to a specified base frame.
   */
  std::vector<scan_point_t> convert_scan(const sensor_msgs::msg::LaserScan::SharedPtr scan, const Matrix<double, 4, 4>& odom_to_base);

  /*
   * Update localization
   */
  void loc_update();

  /*
   * Resets localization to given position.
   */
  void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose);

  /*
   * Stores the given map.
   */
  void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr ros_map);

  /*
   * Extracts a new map tile around current position.
   */
  void update_map();

  /*
   * Asynchronous map update loop, running in separate thread.
   */
  void update_loop();

  /*
   * Publishes "map" frame on tf_->
   */
  void broadcast();

private:
  std::mutex m_node_mutex;

  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::OccupancyGrid>::SharedPtr m_pub_map_tile;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_pub_loc_pose;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_pub_loc_pose_2;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseArray>::SharedPtr m_pub_pose_array;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr m_sub_map_topic;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_sub_scan_topic;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_sub_pose_estimate;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr m_sub_only_use_odom;
  std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

  bool m_broadcast_tf = false;
  bool m_initialized = false;
  bool m_set_initial_pose = false;
  bool m_only_use_odom_enable = false;
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
  bool only_using_odom;

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
  std::atomic<bool> stop_threads_{false};
  std::mutex stop_mtx_;
  std::condition_variable stop_cv_;

  /*
   * @brief Callback executed when a parameter change is detected
   * @param event ParameterEvent message
   */
  rcl_interfaces::msg::SetParametersResult
  dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters);

  // Dynamic parameters handler
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;

  // Dedicated callback group and executor for services and subscriptions in AmclNode,
  // in order to isolate TF timer used in message filter.
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::unique_ptr<nav2_util::NodeThread> executor_thread_;
};

} //namespace neo_localization2

#endif  // NEO_LOCALIZATION2__NEO_LOCALIZATION_NODE_HPP_

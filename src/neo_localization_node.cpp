#include "neo_localization2/neo_localization_node.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;
using nav2::declare_parameter_if_not_declared;
using rcl_interfaces::msg::ParameterType;


namespace neo_localization2
{

NeoLocalizationNode::NeoLocalizationNode(const rclcpp::NodeOptions & options)
: nav2::LifecycleNode("neo_localization", "", options)
{
  RCLCPP_INFO(get_logger(), "Creating");
}

NeoLocalizationNode::~NeoLocalizationNode()
{
  stop_threads_ = true;
  if (m_map_update_thread.joinable()) {
    m_map_update_thread.join();
  }
  if (m_loc_update_timer) {
    m_loc_update_timer->cancel();
    m_loc_update_timer.reset();
  }
}

nav2::CallbackReturn NeoLocalizationNode::on_configure(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(get_logger(), "Configuring");
  auto node = shared_from_this();
  
  callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive, false);
  initParameters(node);
  getParameters(node);
  initTransforms();
  initPubSub();
  initNameSpace();

  if (!dyn_params_handler_) {
    // Add callback for dynamic parameters
    dyn_params_handler_ = this->add_on_set_parameters_callback(
      std::bind(&NeoLocalizationNode::dynamicParametersCallback, this, _1));
  }

  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_callback_group(callback_group_, get_node_base_interface());
  executor_thread_ = std::make_unique<nav2::NodeThread>(executor_);

  return nav2::CallbackReturn::SUCCESS;
}

nav2::CallbackReturn NeoLocalizationNode::on_activate(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(get_logger(), "Activating");

  // Lifecycle publishers must be explicitly activated
  m_pub_map_tile->on_activate();
  m_pub_loc_pose->on_activate();
  m_pub_loc_pose_2->on_activate();
  m_pub_pose_array->on_activate();

  auto node = shared_from_this();

  //Init timer
  if (!m_loc_update_timer) {
    m_loc_update_timer = create_wall_timer(std::chrono::milliseconds(m_loc_update_time_ms), std::bind(&NeoLocalizationNode::loc_update, this));
  } else {
    m_loc_update_timer->reset();
  }

  if (m_map_update_thread.joinable()) {
    stop_threads_ = true;
    stop_cv_.notify_all();
    m_map_update_thread.join();
  }

  // Init map update thread
  stop_threads_ = false;
  m_map_update_thread = std::thread(&NeoLocalizationNode::update_loop, this);

  // create bond connection
  createBond();

  return nav2::CallbackReturn::SUCCESS;
}

nav2::CallbackReturn NeoLocalizationNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Deactivating");

  // stop timer
  if (m_loc_update_timer) {
    m_loc_update_timer->cancel();
    m_loc_update_timer.reset();  
  }

  // ask background loop to stop, but DO NOT join here
  stop_threads_ = true;
  stop_cv_.notify_all();

  if (m_map_update_thread.joinable()) {
    m_map_update_thread.join();
  }
  
  // Now end the bond cleanly
  destroyBond();

  // deactivate lifecycle pubs
  m_pub_map_tile->on_deactivate();
  m_pub_loc_pose->on_deactivate();
  m_pub_loc_pose_2->on_deactivate();
  m_pub_pose_array->on_deactivate();

  return nav2::CallbackReturn::SUCCESS;
}

nav2::CallbackReturn NeoLocalizationNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Cleaning up");

  stop_threads_ = true;
  stop_cv_.notify_all();

  if (m_map_update_thread.joinable()) {
    m_map_update_thread.join();
  }

  // now it’s safe to remove callbacks and free resources
  if (dyn_params_handler_) {
    remove_on_set_parameters_callback(dyn_params_handler_.get());
    dyn_params_handler_.reset();
  }

  map_received_ = false;
  m_initialized = false;
  m_scan_buffer.clear();

  executor_thread_.reset();

  m_map.reset(); 
  m_world.reset();
  transform_listener_.reset();  
  buffer.reset();
  m_tf_broadcaster.reset();

  m_pub_map_tile.reset();
  m_pub_loc_pose.reset();
  m_pub_loc_pose_2.reset();
  m_pub_pose_array.reset();

  m_sub_pose_estimate.reset();
  m_sub_scan_topic.reset();
  m_sub_map_topic.reset();
  m_sub_only_use_odom.reset();


  return nav2::CallbackReturn::SUCCESS;
}


nav2::CallbackReturn NeoLocalizationNode::on_shutdown(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(get_logger(), "Shutting down");
  return nav2::CallbackReturn::SUCCESS;
}

void NeoLocalizationNode::initParameters(nav2::LifecycleNode::SharedPtr node)
{
  // Declare parameters
  declare_parameter_if_not_declared(node, "base_frame", rclcpp::ParameterValue("base_link")); //"Which frame to use for the robot base");
  declare_parameter_if_not_declared(node, "odom_frame", rclcpp::ParameterValue("odom")); //"The name of the odom coordinate frame (local localization)");
  declare_parameter_if_not_declared(node, "map_frame", rclcpp::ParameterValue("map")); //"The name of the coordinate frame published by the localization system");
  declare_parameter_if_not_declared(node, "update_gain", rclcpp::ParameterValue(0.5)); //"Exponential low pass gain for localization update (0 to 1)");
  declare_parameter_if_not_declared(node, "confidence_gain", rclcpp::ParameterValue(0.01)); //"Time based confidence gain when in 2D / 1D mode");
  declare_parameter_if_not_declared(node, "sample_rate", rclcpp::ParameterValue(10)); //"How many particles (samples) to spread (per update)");
  declare_parameter_if_not_declared(node, "loc_update_rate", rclcpp::ParameterValue(100)); //"Localization update rate [ms]");
  declare_parameter_if_not_declared(node, "map_update_rate", rclcpp::ParameterValue(0.5)); //"Map tile update rate [1/s]");
  declare_parameter_if_not_declared(node, "map_size", rclcpp::ParameterValue(1000)); //"Map tile size in pixels");
  declare_parameter_if_not_declared(node, "map_downscale", rclcpp::ParameterValue(0)); //"How often to downscale (half) the original map");
  declare_parameter_if_not_declared(node, "num_smooth", rclcpp::ParameterValue(0)); //"How many 3x3 gaussian smoothing iterations are applied to the map");
  declare_parameter_if_not_declared(node, "min_score", rclcpp::ParameterValue(0.2)); //"Minimum score for valid localization (otherwise 0D mode)");
  declare_parameter_if_not_declared(node, "odometry_std_xy", rclcpp::ParameterValue(0.01)); //"Odometry error in x and y [m/m] (how fast to increase particle spread when in 1D / 0D mode)");
  declare_parameter_if_not_declared(node, "odometry_std_yaw", rclcpp::ParameterValue(0.01)); //"Odometry error in yaw angle [rad/rad] (how fast to increase particle spread when in 0D mode)");
  declare_parameter_if_not_declared(node, "min_sample_std_xy", rclcpp::ParameterValue(0.025)); //"Minimum particle spread in x and y [m]");
  declare_parameter_if_not_declared(node, "min_sample_std_yaw", rclcpp::ParameterValue(0.025)); //"Minimum particle spread in yaw angle [rad]");
  declare_parameter_if_not_declared(node, "max_sample_std_xy", rclcpp::ParameterValue(0.5)); //"Initial/maximum particle spread in x and y [m]");
  declare_parameter_if_not_declared(node, "max_sample_std_yaw", rclcpp::ParameterValue(0.5)); //"Initial/maximum particle spread in yaw angle [rad]");
  declare_parameter_if_not_declared(node, "constrain_threshold", rclcpp::ParameterValue(0.1)); //"Threshold for 1D / 2D position decision making (minimum average second order gradient)");
  declare_parameter_if_not_declared(node, "constrain_threshold_yaw", rclcpp::ParameterValue(0.2)); //"Threshold for 1D / 2D decision making (with or without orientation)");
  declare_parameter_if_not_declared(node, "min_points", rclcpp::ParameterValue(20)); //"Minimum number of points per update");
  declare_parameter_if_not_declared(node, "solver_gain", rclcpp::ParameterValue(0.1)); //"Solver update gain, lower gain = more stability / slower convergence");
  declare_parameter_if_not_declared(node, "solver_damping", rclcpp::ParameterValue(1000.0)); //"Solver update damping, higher damping = more stability / slower convergence");
  declare_parameter_if_not_declared(node, "solver_iterations", rclcpp::ParameterValue(20)); //"Number of gauss-newton iterations per sample per scan");
  declare_parameter_if_not_declared(node, "transform_timeout", rclcpp::ParameterValue(0.2)); //"Maximum wait for getting transforms [s]");
  declare_parameter_if_not_declared(node, "broadcast_tf", rclcpp::ParameterValue(true)); //"Whether broadcast tf or not");
  declare_parameter_if_not_declared(node, "map_topic", rclcpp::ParameterValue("map")); //"Name of map topic");
  declare_parameter_if_not_declared(node, "scan_topic", rclcpp::ParameterValue("scan")); //"Name of scan topic");
  declare_parameter_if_not_declared(node, "initialpose", rclcpp::ParameterValue("initialpose")); //"Name of initial pose topic");
  declare_parameter_if_not_declared(node, "map_tile", rclcpp::ParameterValue("map_tile")); //"Name of map tile topic");
  declare_parameter_if_not_declared(node, "map_pose", rclcpp::ParameterValue("map_pose")); //"Name of map pose topic");
  declare_parameter_if_not_declared(node, "particle_cloud", rclcpp::ParameterValue("particlecloud")); //"Name of particle_cloud topic");
  declare_parameter_if_not_declared(node, "amcl_pose", rclcpp::ParameterValue("amcl_pose")); //"Name of amcl_pose topic");
  declare_parameter_if_not_declared(node, "broadcast_info", rclcpp::ParameterValue(false)); //"Broadcast info for debugging");
  declare_parameter_if_not_declared(node, "set_initial_pose", rclcpp::ParameterValue(true)); //"Whether auto set initial pose or not");
  declare_parameter_if_not_declared(node, "initial_pose.x", rclcpp::ParameterValue(0.0)); //"Initial pose x");
  declare_parameter_if_not_declared(node, "initial_pose.y", rclcpp::ParameterValue(0.0)); //"Initial pose y");
  declare_parameter_if_not_declared(node, "initial_pose.yaw", rclcpp::ParameterValue(0.0)); //"Initial pose yaw");
  declare_parameter_if_not_declared(node, "only_use_odom_enable", rclcpp::ParameterValue(false)); //"Enable a subscriber to choose whether only use odom when localizing or not");
}

void NeoLocalizationNode::getParameters(nav2::LifecycleNode::SharedPtr node)
{
  // Get parameters
  node->get_parameter("base_frame", m_base_frame);
  node->get_parameter("odom_frame", m_odom_frame);
  node->get_parameter("map_frame", m_map_frame);
  node->get_parameter("update_gain", m_update_gain);
  node->get_parameter("confidence_gain", m_confidence_gain);
  node->get_parameter("sample_rate", m_sample_rate);
  node->get_parameter("loc_update_rate", m_loc_update_time_ms);
  node->get_parameter("map_update_rate", m_map_update_rate);
  node->get_parameter("map_size", m_map_size);
  node->get_parameter("map_downscale", m_map_downscale);
  node->get_parameter("num_smooth", m_num_smooth);
  node->get_parameter("min_score", m_min_score);
  node->get_parameter("odometry_std_xy", m_odometry_std_xy);
  node->get_parameter("odometry_std_yaw", m_odometry_std_yaw);
  node->get_parameter("min_sample_std_xy", m_min_sample_std_xy);
  node->get_parameter("min_sample_std_yaw", m_min_sample_std_yaw);
  node->get_parameter("max_sample_std_xy", m_max_sample_std_xy);
  node->get_parameter("max_sample_std_yaw", m_max_sample_std_yaw);
  node->get_parameter("constrain_threshold", m_constrain_threshold);
  node->get_parameter("constrain_threshold_yaw", m_constrain_threshold_yaw);
  node->get_parameter("min_points", m_min_points);
  node->get_parameter("solver_gain", m_solver.gain);
  node->get_parameter("solver_damping", m_solver.damping);
  node->get_parameter("solver_iterations", m_solver_iterations);
  node->get_parameter("transform_timeout", m_transform_timeout);
  node->get_parameter("broadcast_tf", m_broadcast_tf);
  node->get_parameter("map_topic", m_map_topic);
  node->get_parameter("scan_topic", m_scan_topic);
  node->get_parameter("initialpose", m_initial_pose);
  node->get_parameter("map_tile", m_map_tile);
  node->get_parameter("map_pose", m_map_pose);
  node->get_parameter("particle_cloud", m_particle_cloud);
  node->get_parameter("amcl_pose", m_amcl_pose);
  node->get_parameter("broadcast_info", m_broadcast_info);
  node->get_parameter("set_initial_pose", m_set_initial_pose);
  node->get_parameter("initial_pose.x", m_offset_x);
  node->get_parameter("initial_pose.y", m_offset_y);
  node->get_parameter("initial_pose.yaw", m_offset_yaw);
  node->get_parameter("only_use_odom_enable", m_only_use_odom_enable);
}

void NeoLocalizationNode::initTransforms()
{
  m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer);
}

void NeoLocalizationNode::initPubSub()
{
  // Init Subscribers
  m_sub_scan_topic = create_subscription<sensor_msgs::msg::LaserScan>(m_scan_topic, std::bind(&NeoLocalizationNode::scan_callback, this, _1), nav2::qos::SensorDataQoS());
  m_sub_map_topic = create_subscription<nav_msgs::msg::OccupancyGrid>("/map", std::bind(&NeoLocalizationNode::map_callback, this, _1), nav2::qos::LatchedSubscriptionQoS(1));
  m_sub_pose_estimate = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(m_initial_pose, std::bind(&NeoLocalizationNode::pose_callback, this, _1), nav2::qos::StandardTopicQoS(1));
  if (m_only_use_odom_enable) {
    m_sub_only_use_odom = create_subscription<std_msgs::msg::Bool>("/only_use_odom", std::bind(&NeoLocalizationNode::use_odom_callback, this, _1), nav2::qos::LatchedSubscriptionQoS(10));
  }

  // Init Publishers
  m_pub_map_tile = create_publisher<nav_msgs::msg::OccupancyGrid>(m_map_tile, nav2::qos::StandardTopicQoS(1));
  m_pub_loc_pose = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(m_amcl_pose, nav2::qos::LatchedPublisherQoS(1));
  m_pub_loc_pose_2 = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(m_map_pose, nav2::qos::StandardTopicQoS(10));
  m_pub_pose_array = create_publisher<geometry_msgs::msg::PoseArray>(m_particle_cloud, nav2::qos::StandardTopicQoS(10));
}

void NeoLocalizationNode::initNameSpace()
{
  std::string robot_namespace(this->get_namespace());
  // removing the unnecessary "/" from the namespace
  robot_namespace.erase(std::remove(robot_namespace.begin(), robot_namespace.end(), '/'), 
  robot_namespace.end());
  only_using_odom = false;
  m_base_frame = robot_namespace + m_base_frame;
  m_odom_frame = robot_namespace + m_odom_frame;
}

/*
 * Computes localization update for a single laser scan.
 */
void NeoLocalizationNode::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
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
 * Whether to only use odom (no scan_matching) or not
 */
void NeoLocalizationNode::use_odom_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(m_node_mutex);
  if (msg->data == true) {
    only_using_odom = true;
  }
  else {
    only_using_odom = false;
  }
}

/*
 * Convert/Transform a scan from ROS format to a specified base frame.
 */
std::vector<scan_point_t> NeoLocalizationNode::convert_scan(const sensor_msgs::msg::LaserScan::SharedPtr scan, const Matrix<double, 4, 4>& odom_to_base)
{
  std::vector<scan_point_t> points;
  tf2::Stamped<tf2::Transform> base_to_odom;
  tf2::Stamped<tf2::Transform> sensor_to_base;
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

/*
 * Update localization
 */
void NeoLocalizationNode::loc_update()
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
  if(static_cast<int>(points.size()) < m_min_points)
  {
    RCLCPP_WARN_STREAM(this->get_logger(),"NeoLocalizationNode: Number of points too low: " << points.size());
    return;
  }

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header.stamp = tf2_ros::toMsg(base_to_odom.stamp_);
  pose_array.header.frame_id = m_map_frame;

  // calc predicted grid pose based on odometry
  Matrix<double, 3, 1> grid_pose; 
  try {
    grid_pose = (m_grid_to_map.inverse() * T * L * Matrix<double, 4, 1>{0, 0, 0, 1}).project();
  } catch(const std::exception& ex) {
      RCLCPP_WARN_STREAM(this->get_logger(),"NeoLocalizationNode waiting for map updates " << ex.what());
      return;
  }

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
  if ((best_score > m_min_score) && (only_using_odom == false)){
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
        << " rad, mode=" << mode << "D, " << m_scan_buffer.size() << " scans, var_error=" << var_error);
    }
  }

  // clear scan buffer
  m_scan_buffer.clear();
}

/*
  * Resets localization to given position.
  */
void NeoLocalizationNode::pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose)
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
void NeoLocalizationNode::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr ros_map)
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
void NeoLocalizationNode::update_map()
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
  (void)((m_grid_to_map * Matrix<double, 4, 1>{ map->scale() * map->size_x() / 2,
                                  map->scale() * map->size_y() / 2, 0, 1}).project());

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
void NeoLocalizationNode::update_loop()
{
  RCLCPP_INFO_ONCE(this->get_logger(), "NeoLocalizationNode: Activating map update loop");

  // period from Hz (map_update_rate is in 1/s == Hz)
  const auto period = std::chrono::duration<double>(1.0 / std::max(1e-6, m_map_update_rate));

  while (rclcpp::ok() && !stop_threads_) {
    if (!m_initialized && !m_set_initial_pose) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 2000, "Is the robot pose initialized?");
    } else {
      try { update_map(); }
      catch (const std::exception& ex) {
        RCLCPP_WARN(this->get_logger(), "NeoLocalizationNode: update_map() failed: %s", ex.what());
      }
    }

    std::unique_lock<std::mutex> lk(stop_mtx_);
    if (stop_cv_.wait_for(lk, period, [this]{ return stop_threads_.load(); })) {
      break; // woke because we're stopping
    }
  }
}


/*
 * Publishes "map" frame on tf_->
 */
void NeoLocalizationNode::broadcast()
{
  if (!m_broadcast_tf || !m_tf_broadcaster) {
    return;
  }

  geometry_msgs::msg::TransformStamped pose;
  // Use a fresh or normalized time; don’t mutate m_offset_time directly
  auto expiry = rclcpp::Time(m_offset_time) + rclcpp::Duration::from_seconds(1.0);

  pose.header.stamp = expiry;
  pose.header.frame_id = m_map_frame;
  pose.child_frame_id = m_odom_frame;
  pose.transform.translation.x = m_offset_x;
  pose.transform.translation.y = m_offset_y;
  pose.transform.translation.z = 0.0;
  tf2::Quaternion q; q.setRPY(0, 0, m_offset_yaw);
  pose.transform.rotation = tf2::toMsg(q);

  m_tf_broadcaster->sendTransform(pose);
}

/*
 * Dynamic parameters callback
 */
rcl_interfaces::msg::SetParametersResult
NeoLocalizationNode::dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters)
{
  using PT = rcl_interfaces::msg::ParameterType;

  rcl_interfaces::msg::SetParametersResult res;
  res.successful = true;
  res.reason = "ok";

  for (const auto & p : parameters) {
    const std::string & name = p.get_name();
    if (name == "loc_update_rate" || name == "loc_update_rate_ms" || name == "only_use_odom_enable") {
      res.successful = false;
      res.reason = name + ": cannot be changed at runtime";
      return res;
    }
    if (name == "base_frame" || name == "odom_frame" || name == "map_frame" ||
        name == "scan_topic" || name == "map_topic" || name == "initialpose" ||
        name == "map_tile" || name == "map_pose" || name == "particle_cloud" ||
        name == "amcl_pose")
    {
      res.successful = false;
      res.reason = name + ": read-only at runtime";
      return res;
    }
  }

  std::lock_guard<std::mutex> lock(m_node_mutex);

  auto clamp_min = [](double v, double lo) { return v < lo ? lo : v; };
  auto clamp_01  = [](double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); };

  for (const auto & p : parameters) {
    const std::string & name = p.get_name();
    const auto type = p.get_type();

    switch (type) {
      case PT::PARAMETER_DOUBLE: {
        const double v = p.as_double();

        if (name == "update_gain") {
          m_update_gain = clamp_01(v);
        } else if (name == "confidence_gain") {
          m_confidence_gain = clamp_01(v);
        } else if (name == "min_score") {
          m_min_score = clamp_min(v, 0.0);
        } else if (name == "odometry_std_xy") {
          m_odometry_std_xy = clamp_min(v, 0.0);
        } else if (name == "odometry_std_yaw") {
          m_odometry_std_yaw = clamp_min(v, 0.0);
        } else if (name == "min_sample_std_xy") {
          m_min_sample_std_xy = clamp_min(v, 0.0);
          if (m_min_sample_std_xy > m_max_sample_std_xy) {
            m_min_sample_std_xy = m_max_sample_std_xy;
          }
        } else if (name == "min_sample_std_yaw") {
          m_min_sample_std_yaw = clamp_min(v, 0.0);
          if (m_min_sample_std_yaw > m_max_sample_std_yaw) {
            m_min_sample_std_yaw = m_max_sample_std_yaw;
          }
        } else if (name == "max_sample_std_xy") {
          m_max_sample_std_xy = clamp_min(v, 0.0);
          if (m_max_sample_std_xy < m_min_sample_std_xy) {
            m_max_sample_std_xy = m_min_sample_std_xy;
          }
        } else if (name == "max_sample_std_yaw") {
          m_max_sample_std_yaw = clamp_min(v, 0.0);
          if (m_max_sample_std_yaw < m_min_sample_std_yaw) {
            m_max_sample_std_yaw = m_min_sample_std_yaw;
          }
        } else if (name == "constrain_threshold") {
          m_constrain_threshold = clamp_min(v, 0.0);
        } else if (name == "constrain_threshold_yaw") {
          m_constrain_threshold_yaw = clamp_min(v, 0.0);
        } else if (name == "solver_gain") {
          m_solver.gain = clamp_min(v, 0.0);
        } else if (name == "solver_damping") {
          m_solver.damping = clamp_min(v, 0.0);
        } else if (name == "transform_timeout") {
          m_transform_timeout = clamp_min(v, std::numeric_limits<double>::min()); // > 0
        } else if (name == "map_update_rate") {
          m_map_update_rate = clamp_min(v, std::numeric_limits<double>::min()); // > 0 Hz
        } else {
          // ignore unknown doubles (e.g., initial_pose.x/y/yaw if sent here)
        }
        break;
      }

      case PT::PARAMETER_INTEGER: {
        const int64_t vi = p.as_int();
        if (name == "map_size") {
          m_map_size = static_cast<int>(std::max<int64_t>(1, vi));
        } else if (name == "sample_rate") {
          m_sample_rate = static_cast<int>(std::max<int64_t>(1, vi));
        } else if (name == "map_downscale") {
          m_map_downscale = static_cast<int>(std::max<int64_t>(0, vi));
        } else if (name == "num_smooth") {
          m_num_smooth = static_cast<int>(std::max<int64_t>(0, vi));
        } else if (name == "min_points") {
          m_min_points = static_cast<int>(std::max<int64_t>(0, vi));
        } else if (name == "solver_iterations") {
          m_solver_iterations = static_cast<int>(std::max<int64_t>(1, vi));
        } else {
          // ignore unknown ints
        }
        break;
      }

      case PT::PARAMETER_BOOL: {
        const bool vb = p.as_bool();
        if (name == "broadcast_tf") {
          m_broadcast_tf = vb;
        } else if (name == "broadcast_info") {
          m_broadcast_info = vb;
        } else {
          // ignore unknown bools
        }
        break;
      }

      default:
        // reject unsupported types explicitly (prevents silent surprises)
        res.successful = false;
        res.reason = name + ": unsupported parameter type at runtime";
        return res;
    }
  }

  return res;
}


} // namespace neo_localization2

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(neo_localization2::NeoLocalizationNode)


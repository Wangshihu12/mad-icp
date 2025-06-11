// Copyright 2024 R(obots) V(ision) and P(erception) group
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "pipeline.h"

#include <filesystem>
#include <tools/constants.h>

Pipeline::Pipeline(double sensor_hz,
                   bool deskew,
                   double b_max,
                   double rho_ker,
                   double p_th,
                   double b_min,
                   double b_ratio,
                   int num_keyframes,
                   int num_threads,
                   bool realtime) :
  sensor_hz_(sensor_hz),
  deskew_(deskew),
  b_max_(b_max),
  p_th_(p_th),
  b_min_(b_min),
  num_keyframes_(num_keyframes),
  num_threads_(num_threads),
  realtime_(realtime),
  icp_(b_max, rho_ker, b_ratio, num_threads),
  vel_estimator_(sensor_hz) {
  current_tree_ = nullptr;
  frame_to_map_.setIdentity();
  keyframe_to_map_.setIdentity();
  current_velocity_.setZero();
  seq_            = 0;
  seq_keyframe_   = 0;
  is_initialized_ = false;
  is_map_updated_ = false;
  loop_time = (1. / sensor_hz_) * 1000;

  max_parallel_levels_ = static_cast<int>(std::log2(num_threads));
  omp_set_num_threads(num_threads);
}

Pipeline::~Pipeline() {
  while (!frames_.empty()) {
    delete frames_.front()->tree_;
    frames_.pop_front();
  }
  while (!keyframes_.empty()) {
    delete keyframes_.front()->tree_;
    keyframes_.pop_front();
  }
}

void Pipeline::deskew(const ContainerTypePtr& curr_cloud, const Eigen::Isometry3d& T_prev, const Eigen::Isometry3d& T_now) {
  const double ts = 1. / sensor_hz_;

  Vector6d naive_vel;
  Eigen::Isometry3d T_now_to_prev = T_prev.inverse() * T_now;
  naive_vel.head(3)               = T_now_to_prev.translation();
  naive_vel.tail(3)               = logMapSO3(T_now_to_prev.linear());
  naive_vel                       = naive_vel / ts;

  using AzimuthPair = std::pair<double, Eigen::Vector3d>;
  std::vector<AzimuthPair> sorted(curr_cloud->size());

  for (size_t i = 0; i < sorted.size(); ++i) {
    const Eigen::Vector3d& point = curr_cloud->at(i);
    const double azimuth         = atan2(point.y(), point.x());
    sorted[i]                    = std::make_pair(azimuth, point);
  }

  std::sort(
    sorted.begin(), sorted.end(), [](AzimuthPair first, AzimuthPair second) -> bool { return first.first < second.first; });

  const double resolution = 2 * M_PI / double(CHUNKS);
  const double delta      = ts / double(CHUNKS - 1);
  double t                = -ts;

  Vector6d delta_s                     = naive_vel * t;
  Eigen::Isometry3d meas_pose_to_robot = Eigen::Isometry3d::Identity();
  meas_pose_to_robot.linear()          = expMapSO3(delta_s.tail(3));
  meas_pose_to_robot.translation()     = delta_s.head(3);

  double angle = M_PI - resolution;
  for (int i = sorted.size() - 1; i >= 0; --i) {
    if (sorted[i].first < angle) {
      angle -= resolution;
      t += delta;

      delta_s                          = naive_vel * t;
      meas_pose_to_robot               = Eigen::Isometry3d::Identity();
      meas_pose_to_robot.linear()      = expMapSO3(delta_s.tail(3));
      meas_pose_to_robot.translation() = delta_s.head(3);
    }

    (*curr_cloud)[i] = meas_pose_to_robot * sorted[i].second;
  }
}

void Pipeline::compute(const double& curr_stamp, ContainerType curr_cloud_mem) {
  // 获取当前点云数据的指针
  ContainerType* curr_cloud = &curr_cloud_mem;
  // 标记地图未更新状态
  is_map_updated_           = false;

  // 如果管道未初始化，先进行初始化操作
  if (!is_initialized_) {
    initialize(curr_stamp, curr_cloud);
    return;
  }

  // ==================== 预处理阶段时间统计 ====================
  struct timeval preprocessing_start, preprocessing_end, preprocessing_delta;
  gettimeofday(&preprocessing_start, nullptr);

  // 如果启用去畸变且轨迹中有足够的历史数据，对当前点云进行运动去畸变
  // 去畸变可以消除激光雷达扫描过程中由于载体运动造成的点云畸变
  if (deskew_ && trajectory_.size() > 1)
    deskew(curr_cloud, trajectory_[trajectory_.size() - 2], trajectory_[trajectory_.size() - 1]);

  // ==================== 构建当前帧的空间数据结构 ====================
  // 为当前点云构建MAD树（一种适用于点云配准的空间索引树）
  current_tree_ =
    new MADtree(curr_cloud, curr_cloud->begin(), curr_cloud->end(), b_max_, b_min_, 0, max_parallel_levels_, nullptr, nullptr);

  // 清空当前叶子节点列表并重新获取
  current_leaves_.clear();
  current_tree_->getLeafs(std::back_insert_iterator<LeafList>(current_leaves_));

  // ==================== 运动预测阶段 ====================
  // 基于当前估计的速度计算时间步长内的运动增量
  Vector6d dx = current_velocity_ * 1. / sensor_hz_;  // dx包含平移和旋转增量
  Eigen::Isometry3d dX;
  const Eigen::Matrix3d dR = expMapSO3(dx.tail(3));  // 将旋转向量转换为旋转矩阵
  dX.setIdentity();
  dX.linear()                  = dR;                  // 设置旋转部分
  dX.translation()             = dx.head(3);          // 设置平移部分
  // 计算当前帧相对于地图的预测位姿
  Eigen::Isometry3d prediction = frame_to_map_ * dX;

  // ==================== ICP配准初始化 ====================
  // 设置ICP算法的移动点云（当前帧的叶子节点）
  icp_.setMoving(current_leaves_);
  // 用预测位姿初始化ICP算法
  icp_.init(prediction);

  // ICP迭代时间统计变量
  float icp_time       = 0;
  float total_icp_time = 0;

  // 计算预处理阶段耗时
  gettimeofday(&preprocessing_end, nullptr);
  timersub(&preprocessing_end, &preprocessing_start, &preprocessing_delta);
  const float preprocessing_time = float(preprocessing_delta.tv_sec) * 1000. + 1e-3 * preprocessing_delta.tv_usec;

  struct timeval icp_start, icp_end, icp_delta;

  // ==================== ICP迭代配准阶段 ====================
  for (size_t icp_iteration = 0; icp_iteration < MAX_ICP_ITS; ++icp_iteration) {
    // 实时模式下检查剩余时间，如果时间不足则提前退出
    const float remaining_time = loop_time - 5.0 - (preprocessing_time + total_icp_time + icp_time);
    if (realtime_ && remaining_time < 0)
      break;

    gettimeofday(&icp_start, nullptr);
    
    // 在最后一次迭代中，重置所有叶子节点的匹配状态
    if (icp_iteration == MAX_ICP_ITS - 1) {
      for (MADtree* l : current_leaves_) {
        l->matched_ = false;
      }
    }

    // 重置ICP的累加器
    icp_.resetAdders();

    // ==================== 并行处理关键帧 ====================
    // 使用OpenMP并行处理所有关键帧，更新ICP状态
#pragma omp parallel for
    for (const Frame* frame : keyframes_) {
      icp_.update(frame->tree_);
    }

#pragma omp barrier  // 等待所有线程完成

    // 更新ICP的状态（计算新的变换矩阵）
    icp_.updateState();

    // 统计本次ICP迭代耗时
    gettimeofday(&icp_end, nullptr);
    timersub(&icp_end, &icp_start, &icp_delta);
    icp_time = float(icp_delta.tv_sec) * 1000. + 1e-3 * icp_delta.tv_usec;
    total_icp_time += icp_time;
  }

  // ==================== 更新位姿和统计信息 ====================
  // 更新当前帧到地图的变换矩阵
  frame_to_map_ = icp_.X_;

  // 统计匹配成功的叶子节点数量
  int matched_leaves = 0;
  for (MADtree* l : current_leaves_) {
    if (l->matched_) {
      matched_leaves++;
    }
  }

  // 计算内点比例（匹配成功的点的比例）
  double inliers_ratio = double(matched_leaves) / double(current_leaves_.size());

  // ==================== 轨迹和速度更新 ====================
  // 将当前位姿添加到轨迹中
  trajectory_.push_back(frame_to_map_);

  // 构建用于速度估计的里程计窗口
  std::vector<Eigen::Isometry3d> odom_window;
  for (int i = std::max(0, int(trajectory_.size()) - SMOOTHING_T); i < trajectory_.size(); ++i) {
    odom_window.push_back(trajectory_[i]);
  }

  // 使用速度估计器更新当前速度
  vel_estimator_.init(current_velocity_);
  vel_estimator_.setOdometry(odom_window);
  vel_estimator_.oneRound();
  current_velocity_ = vel_estimator_.X_;

  // ==================== 帧管理 ====================
  // 创建当前帧对象并设置相关属性
  Frame* current_frame(new Frame);
  current_frame->frame_        = seq_;                                    // 帧序号
  current_frame->frame_to_map_ = frame_to_map_;                          // 帧到地图的变换
  current_frame->stamp_        = curr_stamp;                             // 时间戳
  current_frame->weight_       = icp_.H_adder_.inverse().determinant();  // 权重（基于海塞矩阵）
  
  // 将变换应用到当前树结构
  current_tree_->applyTransform(frame_to_map_.linear(), frame_to_map_.translation());
  current_frame->tree_   = current_tree_;
  current_frame->leaves_ = current_leaves_;

  // 将当前帧添加到帧队列中
  frames_.push_back(current_frame);
  // 如果帧队列超过窗口大小，删除最旧的帧
  if (frames_.size() > FRAME_WINDOW) {
    delete frames_.front()->tree_;
    frames_.pop_front();
  }

  // ==================== 关键帧选择策略 ====================
  // 如果内点比例低于阈值，说明当前环境变化较大，需要选择新的关键帧
  if (inliers_ratio < p_th_) {
    // 在当前帧窗口中寻找权重最小（最可靠）的帧作为关键帧
    double best_weight = std::numeric_limits<double>::max();
    int new_seq        = 0;
    Frame* best_frame  = nullptr;
    for (Frame* frame : frames_) {
      if (frame->weight_ < best_weight) {
        best_weight = frame->weight_;
        new_seq     = frame->frame_;
        best_frame  = frame;
      }
    }

    // 清理帧队列，删除选中关键帧之前的所有帧
    while (!frames_.empty() && frames_.front()->frame_ <= new_seq) {
      if (frames_.front()->frame_ < new_seq) {
        delete frames_.front()->tree_;
      }
      frames_.pop_front();
    }

    // 将选中的帧添加到关键帧队列
    keyframes_.push_back(best_frame);
    // 如果关键帧数量超过限制，删除最旧的关键帧
    if (keyframes_.size() > num_keyframes_) {
      delete keyframes_.front()->tree_;
      keyframes_.pop_front();
    }

    // 更新地图状态和关键帧信息
    is_map_updated_  = true;
    seq_keyframe_    = new_seq;
    keyframe_to_map_ = best_frame->frame_to_map_;
  }

  // 递增帧序号
  seq_++;
}

void Pipeline::initialize(const double& curr_stamp, const ContainerTypePtr curr_cloud) {
  Frame* current_frame(new Frame);
  current_frame->frame_        = seq_;
  current_frame->frame_to_map_ = frame_to_map_;
  current_frame->stamp_        = curr_stamp;
  current_frame->tree_ =
    new MADtree(curr_cloud, curr_cloud->begin(), curr_cloud->end(), b_max_, b_min_, 0, max_parallel_levels_, nullptr, nullptr);

  current_frame->tree_->getLeafs(std::back_insert_iterator<LeafList>(current_frame->leaves_));

  keyframes_.push_back(current_frame);

  trajectory_.push_back(Eigen::Isometry3d::Identity());

  is_initialized_ = true;
  is_map_updated_ = true;
  seq_++;
}

const bool Pipeline::isMapUpdated() {
  return is_map_updated_;
}

const ContainerType Pipeline::currentLeaves() {
  ContainerType leaves;
  std::back_insert_iterator<ContainerType> leaves_it(leaves);
  for (MADtree* leaf : current_leaves_) {
    ++leaves_it = leaf->mean_;
  }
  return leaves;
}

const ContainerType Pipeline::modelLeaves() {
  ContainerType leaves;
  std::back_insert_iterator<ContainerType> leaves_it(leaves);
  for (auto frame : keyframes_) {
    for (MADtree* leaf : frame->leaves_) {
      ++leaves_it = leaf->mean_;
    }
  }
  return leaves;
}
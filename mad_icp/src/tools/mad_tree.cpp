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

#include "mad_tree.h"

#include <Eigen/Eigenvalues>
#include <fstream>
#include <future>

MADtree::MADtree(const ContainerTypePtr vec,
                 const IteratorType begin,
                 const IteratorType end,
                 const double b_max,
                 const double b_min,
                 const int level,
                 const int max_parallel_level,
                 MADtree* parent,
                 MADtree* plane_predecessor) {
  build(vec, begin, end, b_max, b_min, level, max_parallel_level, parent, plane_predecessor);
}

void MADtree::build(const ContainerTypePtr vec,
                    const IteratorType begin,
                    const IteratorType end,
                    const double b_max,
                    const double b_min,
                    const int level,
                    const int max_parallel_level,
                    MADtree* parent,
                    MADtree* plane_predecessor) {
  // ==================== 初始化节点属性 ====================
  // 设置当前节点的父节点指针
  parent_ = parent;
  
  // ==================== 统计分析阶段 ====================
  Eigen::Matrix3d cov;
  // 计算当前点集的均值和协方差矩阵
  // 均值表示点云的中心位置，协方差矩阵描述点云的分布特征
  computeMeanAndCovariance(mean_, cov, begin, end);
  
  // 对协方差矩阵进行特征值分解
  // 特征向量表示点云的主要分布方向（主成分方向）
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
  es.computeDirect(cov);
  eigenvectors_ = es.eigenvectors();
  
  // 计算在特征向量坐标系下的轴对齐包围盒
  // bbox_存储三个主方向上的尺寸，用于判断点云的几何特征
  num_points_   = computeBoundingBox(bbox_, mean_, eigenvectors_.transpose(), begin, end);

  // ==================== 叶节点判断和处理 ====================
  // 如果最小方向（通常是厚度方向）的尺寸小于b_max，说明点云呈现平面特征
  // 此时创建叶节点而不再继续分割
  if (bbox_(2) < b_max) {
    // 如果存在平面前驱节点，继承其第一主方向（法向量方向）
    // 这样可以保持平面法向量的一致性
    if (plane_predecessor) {
      eigenvectors_.col(0) = plane_predecessor->eigenvectors_.col(0);
    } else {
      // 如果当前节点点数太少（<3），向上查找父节点来获取稳定的法向量
      // 这样可以避免少量点计算出的不稳定法向量
      if (num_points_ < 3) {
        MADtree* node = this;
        while (node->parent_ && node->num_points_ < 3)
          node = node->parent_;
        eigenvectors_.col(0) = node->eigenvectors_.col(0);
      }
    }

    // ==================== 寻找最近点作为代表点 ====================
    // 对于叶节点，选择距离均值最近的点作为该节点的代表点
    // 这样可以减少量化误差，提高匹配精度
    Eigen::Vector3d& nearest_point = *begin;
    double shortest_dist           = std::numeric_limits<double>::max();
    for (IteratorType it = begin; it != end; ++it) {
      const Eigen::Vector3d& v = *it;
      const double dist        = (v - mean_).norm();
      if (dist < shortest_dist) {
        nearest_point = v;
        shortest_dist = dist;
      }
    }
    // 用最近点替换均值作为节点的代表位置
    mean_ = nearest_point;

    // 叶节点处理完成，直接返回
    return;
  }
  
  // ==================== 平面前驱节点更新 ====================
  // 如果当前节点还没有平面前驱，且第一主方向尺寸小于b_min
  // 说明当前节点开始表现出平面特征，将其设为后续子节点的平面前驱
  if (!plane_predecessor) {
    if (bbox_(0) < b_min)
      plane_predecessor = this;
  }

  // ==================== 节点分割阶段 ====================
  // 使用第三主方向（最大变化方向）的特征向量作为分割平面的法向量
  // 这样可以最大化地分离点云，提高树的平衡性
  const Eigen::Vector3d& _split_plane_normal = eigenvectors_.col(2);
  
  // 根据分割平面将点云分为两部分
  // 分割准则：点到均值的向量在分割平面法向量上的投影小于0的点归为左子树
  IteratorType middle =
    split(begin, end, [&](const Eigen::Vector3d& p) -> bool { return (p - mean_).dot(_split_plane_normal) < double(0); });

  // ==================== 子树构建阶段 ====================
  // 根据当前层级决定是否使用并行构建
  if (level >= max_parallel_level) {
    // 串行构建：当层级较深时，避免创建过多线程导致开销过大
    left_ =
      new MADtree(vec, begin, middle, b_max, b_min, level + 1, max_parallel_level, this, plane_predecessor);

    right_ =
      new MADtree(vec, middle, end, b_max, b_min, level + 1, max_parallel_level, this, plane_predecessor);
  } else {
    // 并行构建：在较浅层级使用异步任务并行构建左右子树
    // 这样可以充分利用多核处理器，加速树的构建过程
    std::future<MADtree*> l = std::async(MADtree::makeSubtree,
                                         vec,
                                         begin,
                                         middle,
                                         b_max,
                                         b_min,
                                         level + 1,
                                         max_parallel_level,
                                         this,
                                         plane_predecessor);

    std::future<MADtree*> r = std::async(MADtree::makeSubtree,
                                         vec,
                                         middle,
                                         end,
                                         b_max,
                                         b_min,
                                         level + 1,
                                         max_parallel_level,
                                         this,
                                         plane_predecessor);
    // 等待两个子树构建完成
    left_                   = l.get();
    right_                  = r.get();
  }
}

MADtree* MADtree::makeSubtree(const ContainerTypePtr vec,
                              const IteratorType begin,
                              const IteratorType end,
                              const double b_max,
                              const double b_min,
                              const int level,
                              const int max_parallel_level,
                              MADtree* parent,
                              MADtree* plane_predecessor) {
  return new MADtree(vec, begin, end, b_max, b_min, level, max_parallel_level, parent, plane_predecessor);
}

const MADtree* MADtree::bestMatchingLeafFast(const Eigen::Vector3d& query) const {
  const MADtree* node = this;
  while (node->left_ || node->right_) {
    const Eigen::Vector3d& _split_plane_normal = node->eigenvectors_.col(2);
    node = ((query - node->mean_).dot(_split_plane_normal) < double(0)) ? node->left_ : node->right_;
  }

  return node;
}

void MADtree::getLeafs(std::back_insert_iterator<std::vector<MADtree*>> it) {
  if (!left_ && !right_) {
    ++it = this;
    return;
  }
  if (left_)
    left_->getLeafs(it);
  if (right_)
    right_->getLeafs(it);
}

void MADtree::applyTransform(const Eigen::Matrix3d& r, const Eigen::Vector3d& t) {
  mean_         = r * mean_ + t;
  eigenvectors_ = r * eigenvectors_;
  if (left_)
    left_->applyTransform(r, t);
  if (right_)
    right_->applyTransform(r, t);
}

// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/scheduling/agent_generated_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

int64_t NodeBitCount(Node* node) {
  return node->GetType()->GetFlatBitCount();
}

int64_t NodeFanout(Node* node) {
  return std::max<int64_t>(1, node->users().size());
}

}  // namespace

namespace {

int64_t AgentListAbs(int64_t x) { return x < 0 ? -x : x; }

int64_t AgentListNodeDelayPsOrZero(const DelayEstimator& delay_estimator,
                                   Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentListSameStageStartTime(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps) {
  int64_t start_time_ps = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto cycle_it = assigned_cycles.find(operand);
    if (cycle_it == assigned_cycles.end() || cycle_it->second != candidate_cycle) {
      continue;
    }
    auto time_it = completion_time_ps.find(operand);
    if (time_it != completion_time_ps.end()) {
      start_time_ps = std::max(start_time_ps, time_it->second);
    }
  }
  return start_time_ps;
}

int64_t AgentListOperandBoundaryCost(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t cost = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    if (it->second < candidate_cycle) {
      cost += NodeBitCount(operand) * (candidate_cycle - it->second);
    }
  }
  return cost;
}

int64_t AgentListProjectedUserBoundaryCost(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    sched::ScheduleBounds* bounds) {
  int64_t cost = 0;
  const int64_t node_bits = NodeBitCount(node);
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    if (it != assigned_cycles.end()) {
      if (candidate_cycle < it->second) {
        cost += node_bits * (it->second - candidate_cycle);
      }
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    if (candidate_cycle < user_lb) {
      cost += node_bits * (user_lb - candidate_cycle);
    }
  }
  return cost;
}

int64_t AgentListForcedLaterUserCost(Node* node, int64_t candidate_cycle,
                                     sched::ScheduleBounds* bounds) {
  int64_t cost = 0;
  const int64_t node_bits = NodeBitCount(node);
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    const int64_t user_ub = bounds->ub(user);
    if (candidate_cycle > user_lb) {
      const int64_t delta = candidate_cycle - user_lb;
      const int64_t mobility = std::max<int64_t>(0, user_ub - user_lb);
      cost += delta * node_bits * std::max<int64_t>(1, NodeFanout(user));
      cost += (delta * std::max<int64_t>(1, NodeBitCount(user))) /
              std::max<int64_t>(1, mobility + 1);
    }
  }
  return cost;
}

int64_t AgentListSuccessorUrgency(Node* node, sched::ScheduleBounds* bounds) {
  int64_t urgency = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    const int64_t mobility = std::max<int64_t>(0, bounds->ub(user) - bounds->lb(user));
    urgency += std::max<int64_t>(1, NodeFanout(user)) *
               std::max<int64_t>(1, NodeBitCount(user) / 8 + 1) /
               std::max<int64_t>(1, mobility + 1);
  }
  return std::max<int64_t>(1, urgency);
}

int64_t AgentListStageDensityCost(
    int64_t cycle,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count) {
  auto it = stage_node_count.find(cycle);
  return it == stage_node_count.end() ? 0 : it->second;
}

int64_t AgentListScoreCandidate(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t clock_period_ps, int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count,
    sched::ScheduleBounds* bounds) {
  const int64_t start_time_ps = AgentListSameStageStartTime(
      node, candidate_cycle, assigned_cycles, completion_time_ps);
  const int64_t end_time_ps = start_time_ps + node_delay_ps;
  const int64_t timing_overflow_ps =
      std::max<int64_t>(0, end_time_ps - clock_period_ps);

  const int64_t operand_boundary_cost =
      AgentListOperandBoundaryCost(node, candidate_cycle, assigned_cycles);
  const int64_t projected_user_boundary_cost =
      AgentListProjectedUserBoundaryCost(node, candidate_cycle, assigned_cycles,
                                         bounds);
  const int64_t forced_later_user_cost =
      AgentListForcedLaterUserCost(node, candidate_cycle, bounds);
  const int64_t stage_density_cost =
      AgentListStageDensityCost(candidate_cycle, stage_node_count);

  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t lateness = candidate_cycle - lb;
  const int64_t midpoint = (lb + ub) / 2;
  const int64_t center_bias = AgentListAbs(candidate_cycle - midpoint);
  const int64_t criticality =
      std::max<int64_t>(1, NodeBitCount(node) / 8 + NodeFanout(node));
  const int64_t successor_urgency = AgentListSuccessorUrgency(node, bounds);

  return timing_overflow_ps * timing_overflow_ps * 1000000000LL +
         operand_boundary_cost * 8192LL +
         projected_user_boundary_cost * 4096LL +
         forced_later_user_cost * 16384LL +
         stage_density_cost * 32LL +
         (lateness * criticality * successor_urgency * 8LL) /
             std::max<int64_t>(1, mobility + 1) +
         center_bias;
}

}  // namespace

namespace {

int64_t AgentDpClamp(int64_t value, int64_t lo, int64_t hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

int64_t AgentDpNodeDelayPsOrZero(const DelayEstimator& delay_estimator,
                                 Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentDpEarliestFromAssignedOperands(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    sched::ScheduleBounds* bounds) {
  int64_t lb = bounds->lb(node);
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it != assigned_cycles.end()) {
      lb = std::max(lb, it->second);
    }
  }
  return lb;
}

int64_t AgentDpSinkDepth(Node* node,
                         const absl::flat_hash_map<Node*, int64_t>& sink_depth) {
  auto it = sink_depth.find(node);
  return it == sink_depth.end() ? 0 : it->second;
}

int64_t AgentDpComputeStageQuota(int64_t timed_node_count,
                                 int64_t pipeline_stages, int64_t cycle) {
  if (pipeline_stages <= 0) {
    return timed_node_count;
  }
  const int64_t base = timed_node_count / pipeline_stages;
  const int64_t rem = timed_node_count % pipeline_stages;
  return base + (cycle < rem ? 1 : 0);
}

int64_t AgentDpRecomputeNodeCost(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t cost = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    const int64_t delta = assigned_cycles.at(node) - it->second;
    if (delta > 0) {
      cost += delta * NodeBitCount(operand);
    }
  }
  return cost;
}

}  // namespace

namespace {

int64_t AgentHybridAbs(int64_t x) { return x < 0 ? -x : x; }

int64_t AgentHybridDelayPsOrZero(const DelayEstimator& delay_estimator,
                                 Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentHybridLatestAssignedOperandCycle(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t latest = 0;
  bool any = false;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    latest = any ? std::max(latest, it->second) : it->second;
    any = true;
  }
  return any ? latest : 0;
}

int64_t AgentHybridEarliestLiveUserLb(Node* node, sched::ScheduleBounds* bounds) {
  int64_t best = std::numeric_limits<int64_t>::max();
  bool any = false;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    best = any ? std::min(best, bounds->lb(user)) : bounds->lb(user);
    any = true;
  }
  return any ? best : bounds->ub(node);
}

int64_t AgentHybridSinkDistance(Node* node,
                                const absl::flat_hash_map<Node*, int64_t>&
                                    reverse_depth) {
  auto it = reverse_depth.find(node);
  return it == reverse_depth.end() ? 0 : it->second;
}

int64_t AgentHybridTimedUserCount(Node* node) {
  int64_t count = 0;
  for (Node* user : node->users()) {
    if (!IsUntimed(user)) {
      ++count;
    }
  }
  return std::max<int64_t>(count, 1);
}

int64_t AgentHybridCandidateScore(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t pipeline_stages, int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& reverse_depth,
    const std::vector<int64_t>& stage_load,
    const std::vector<int64_t>& stage_bits,
    sched::ScheduleBounds* bounds) {
  const int64_t bits = NodeBitCount(node);
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t timed_users = AgentHybridTimedUserCount(node);
  const int64_t sink_depth = AgentHybridSinkDistance(node, reverse_depth);
  const int64_t latest_operand_cycle =
      AgentHybridLatestAssignedOperandCycle(node, assigned_cycles);
  const int64_t earliest_user_lb = AgentHybridEarliestLiveUserLb(node, bounds);

  int64_t operand_reg_cost = 0;
  int64_t operand_same_stage_bonus = 0;
  int64_t operand_stage_span = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    const int64_t pred_cycle = it->second;
    if (candidate_cycle > pred_cycle) {
      operand_reg_cost += NodeBitCount(operand) * (candidate_cycle - pred_cycle);
      operand_stage_span += candidate_cycle - pred_cycle;
    } else if (candidate_cycle == pred_cycle) {
      operand_same_stage_bonus += NodeBitCount(operand);
    }
  }

  int64_t user_pressure_cost = 0;
  int64_t fixed_user_pull = 0;
  int64_t flexible_user_pull = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    const int64_t user_bits = NodeBitCount(user);
    auto assigned_it = assigned_cycles.find(user);
    if (assigned_it != assigned_cycles.end()) {
      const int64_t user_cycle = assigned_it->second;
      if (user_cycle > candidate_cycle) {
        fixed_user_pull += bits * (user_cycle - candidate_cycle);
      }
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    const int64_t user_ub = bounds->ub(user);
    const int64_t user_mobility = std::max<int64_t>(0, user_ub - user_lb);
    if (candidate_cycle > user_lb) {
      const int64_t delta = candidate_cycle - user_lb;
      user_pressure_cost +=
          delta * bits *
          (std::max<int64_t>(1, NodeFanout(user)) + std::max<int64_t>(1, user_bits / 16));
      user_pressure_cost +=
          (delta * std::max<int64_t>(1, bits + user_bits / 4)) /
          std::max<int64_t>(1, user_mobility + 1);
    } else if (candidate_cycle < user_lb) {
      flexible_user_pull += bits * (user_lb - candidate_cycle);
    }
  }

  int64_t stage_balance_cost = 0;
  if (candidate_cycle >= 0 &&
      candidate_cycle < static_cast<int64_t>(stage_load.size())) {
    const int64_t total_assigned =
        assigned_cycles.size() > std::numeric_limits<int64_t>::max() / 2
            ? std::numeric_limits<int64_t>::max() / 2
            : static_cast<int64_t>(assigned_cycles.size());
    const int64_t expected =
        pipeline_stages > 0 ? total_assigned / pipeline_stages : 0;
    stage_balance_cost += AgentHybridAbs(stage_load[candidate_cycle] - expected);
    stage_balance_cost += stage_bits[candidate_cycle] / 64;
    if (candidate_cycle > 0 &&
        candidate_cycle - 1 < static_cast<int64_t>(stage_load.size())) {
      stage_balance_cost +=
          AgentHybridAbs((stage_load[candidate_cycle] + 1) -
                         stage_load[candidate_cycle - 1]);
    }
    if (candidate_cycle + 1 < static_cast<int64_t>(stage_load.size())) {
      stage_balance_cost +=
          AgentHybridAbs((stage_load[candidate_cycle] + 1) -
                         stage_load[candidate_cycle + 1]);
    }
  }

  const int64_t center = pipeline_stages > 0 ? (pipeline_stages - 1) / 2 : 0;
  const int64_t spread_bias =
      AgentHybridAbs(candidate_cycle - center) *
      std::max<int64_t>(1, bits / 32 + timed_users);
  const int64_t asap_bias =
      (candidate_cycle - lb) *
      std::max<int64_t>(1, (sink_depth + timed_users) / std::max<int64_t>(1, mobility + 1));
  const int64_t alap_bias =
      std::max<int64_t>(0, earliest_user_lb - candidate_cycle) *
      std::max<int64_t>(1, bits / 32);
  const int64_t timing_chain_bias =
      (candidate_cycle == latest_operand_cycle ? 0 : 1) * (node_delay_ps > 0 ? 1 : 0);

  return operand_reg_cost * 4096LL +
         user_pressure_cost * 8192LL +
         fixed_user_pull * 2048LL +
         flexible_user_pull * 1024LL +
         stage_balance_cost * 64LL +
         asap_bias * 256LL +
         alap_bias * 32LL +
         spread_bias * 8LL +
         operand_stage_span * 64LL +
         timing_chain_bias * 4LL -
         operand_same_stage_bonus * 16LL;
}

}  // namespace

namespace {

int64_t AgentAsapDelayPsOrZero(const DelayEstimator& delay_estimator,
                               Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentAsapSameStageSlackScore(
    Node* node, int64_t candidate_cycle, int64_t clock_period_ps,
    int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t same_stage_operand_delay_ps = 0;
  int64_t same_stage_operand_bits = 0;
  int64_t boundary_bits = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    if (it->second == candidate_cycle) {
      same_stage_operand_delay_ps += AgentAsapDelayPsOrZero(
          *static_cast<const DelayEstimator*>(nullptr), operand);
      same_stage_operand_bits += NodeBitCount(operand);
    } else if (it->second < candidate_cycle) {
      boundary_bits += NodeBitCount(operand) * (candidate_cycle - it->second);
    }
  }
  const int64_t slack_ps =
      clock_period_ps - (same_stage_operand_delay_ps + node_delay_ps);
  return slack_ps * 1024LL + same_stage_operand_bits - boundary_bits;
}

int64_t AgentAsapTieBreakScore(
    Node* node, int64_t candidate_cycle, int64_t clock_period_ps,
    int64_t node_delay_ps, const DelayEstimator& delay_estimator,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t same_stage_operand_delay_ps = 0;
  int64_t same_stage_operand_bits = 0;
  int64_t cross_stage_bits = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    if (it->second == candidate_cycle) {
      same_stage_operand_delay_ps +=
          AgentAsapDelayPsOrZero(delay_estimator, operand);
      same_stage_operand_bits += NodeBitCount(operand);
    } else if (it->second < candidate_cycle) {
      cross_stage_bits += NodeBitCount(operand) * (candidate_cycle - it->second);
    }
  }
  const int64_t slack_ps =
      clock_period_ps - (same_stage_operand_delay_ps + node_delay_ps);
  return slack_ps * 4096LL + same_stage_operand_bits * 4LL - cross_stage_bits;
}

}  // namespace

namespace {

int64_t AgentDpStageTarget(int64_t total_timed_nodes, int64_t pipeline_stages,
                           int64_t cycle) {
  if (pipeline_stages <= 0) {
    return total_timed_nodes;
  }
  const int64_t base = total_timed_nodes / pipeline_stages;
  const int64_t rem = total_timed_nodes % pipeline_stages;
  return base + (cycle < rem ? 1 : 0);
}

int64_t AgentDpReverseDepth(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& reverse_depth) {
  auto it = reverse_depth.find(node);
  return it == reverse_depth.end() ? 0 : it->second;
}

int64_t AgentDpTimedUsers(Node* node) {
  int64_t count = 0;
  for (Node* user : node->users()) {
    if (!IsUntimed(user)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentDpCandidateCost(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t pipeline_stages,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& reverse_depth,
    const std::vector<int64_t>& stage_load) {
  int64_t reg_cost = 0;
  int64_t same_stage_bonus = 0;
  int64_t pred_count = 0;
  int64_t latest_pred_cycle = lb;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    ++pred_count;
    latest_pred_cycle = std::max(latest_pred_cycle, it->second);
    const int64_t delta = candidate_cycle - it->second;
    if (delta > 0) {
      reg_cost += delta * NodeBitCount(operand);
    } else if (delta == 0) {
      same_stage_bonus += NodeBitCount(operand);
    }
  }

  const int64_t bits = NodeBitCount(node);
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t sink_depth = AgentDpReverseDepth(node, reverse_depth);
  const int64_t fanout = AgentDpTimedUsers(node);

  int64_t future_pull_cost = 0;
  int64_t future_lock_cost = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    if (it != assigned_cycles.end()) {
      if (it->second > candidate_cycle) {
        future_pull_cost += bits * (it->second - candidate_cycle);
      }
      continue;
    }
    const int64_t user_lb = lb <= ub ? std::max<int64_t>(0, lb) : 0;
    (void)user_lb;
    const int64_t implied_gap = std::max<int64_t>(0, candidate_cycle - lb);
    if (implied_gap > 0) {
      future_lock_cost += implied_gap * bits;
    }
  }

  int64_t balance_cost = 0;
  if (candidate_cycle >= 0 &&
      candidate_cycle < static_cast<int64_t>(stage_load.size())) {
    const int64_t target =
        AgentDpStageTarget(static_cast<int64_t>(assigned_cycles.size()) + 1,
                           pipeline_stages, candidate_cycle);
    balance_cost += std::max<int64_t>(0, stage_load[candidate_cycle] + 1 - target);
    if (candidate_cycle > 0) {
      balance_cost += std::max<int64_t>(
          0, (stage_load[candidate_cycle] + 1) - stage_load[candidate_cycle - 1] - 1);
    }
  }

  const int64_t center = pipeline_stages > 0 ? (pipeline_stages - 1) / 2 : 0;
  const int64_t spread_bias =
      std::abs(candidate_cycle - center) * std::max<int64_t>(1, bits / 32);
  const int64_t lateness_bias =
      (candidate_cycle - lb) *
      std::max<int64_t>(1, (sink_depth + fanout) / (mobility + 1));
  const int64_t chain_bonus =
      (candidate_cycle == latest_pred_cycle ? 1 : 0) *
      std::max<int64_t>(1, pred_count);

  return reg_cost * 4096LL + future_pull_cost * 1024LL +
         future_lock_cost * 256LL + balance_cost * 64LL +
         lateness_bias * 32LL + spread_bias -
         same_stage_bonus * 16LL - chain_bonus * 8LL;
}

}  // namespace

namespace {

int64_t AgentSchedDelayPsOrZero(const DelayEstimator& delay_estimator,
                                Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentSchedTimedUserCount(Node* node) {
  int64_t count = 0;
  for (Node* user : node->users()) {
    if (!IsUntimed(user)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentSchedTimedOperandCount(Node* node) {
  int64_t count = 0;
  for (Node* operand : node->operands()) {
    if (!IsUntimed(operand)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentSchedReverseDepth(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& reverse_depth) {
  auto it = reverse_depth.find(node);
  return it == reverse_depth.end() ? 0 : it->second;
}

int64_t AgentSchedLatestAssignedOperandCycle(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    int64_t fallback) {
  int64_t latest = fallback;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it != assigned_cycles.end()) {
      latest = std::max(latest, it->second);
    }
  }
  return latest;
}

int64_t AgentSchedCandidateScore(
    Node* node, int64_t cycle, int64_t lb, int64_t ub, int64_t pipeline_stages,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& reverse_depth,
    const std::vector<int64_t>& stage_load,
    const std::vector<int64_t>& stage_bit_load, sched::ScheduleBounds* bounds) {
  const int64_t bits = NodeBitCount(node);
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t reverse = AgentSchedReverseDepth(node, reverse_depth);
  const int64_t fanout = std::max<int64_t>(1, AgentSchedTimedUserCount(node));
  const int64_t fanin = std::max<int64_t>(1, AgentSchedTimedOperandCount(node));
  const int64_t latest_operand_cycle =
      AgentSchedLatestAssignedOperandCycle(node, assigned_cycles, lb);

  int64_t operand_boundary_cost = 0;
  int64_t operand_same_stage_bits = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    if (it->second < cycle) {
      operand_boundary_cost += NodeBitCount(operand) * (cycle - it->second);
    } else if (it->second == cycle) {
      operand_same_stage_bits += NodeBitCount(operand);
    }
  }

  int64_t user_boundary_cost = 0;
  int64_t user_forcing_cost = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    if (it != assigned_cycles.end()) {
      if (cycle < it->second) {
        user_boundary_cost += bits * (it->second - cycle);
      }
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    const int64_t user_ub = std::min<int64_t>(bounds->ub(user), pipeline_stages - 1);
    const int64_t user_mobility = std::max<int64_t>(0, user_ub - user_lb);
    if (cycle < user_lb) {
      user_boundary_cost += bits * (user_lb - cycle);
    } else if (cycle > user_lb) {
      const int64_t delta = cycle - user_lb;
      user_forcing_cost +=
          delta * bits *
          (std::max<int64_t>(1, NodeFanout(user)) +
           std::max<int64_t>(1, NodeBitCount(user) / 16)) /
          std::max<int64_t>(1, user_mobility + 1);
    }
  }

  int64_t balance_cost = 0;
  if (cycle >= 0 && cycle < static_cast<int64_t>(stage_load.size())) {
    const int64_t total_scheduled = static_cast<int64_t>(assigned_cycles.size());
    const int64_t ideal_load =
        pipeline_stages > 0 ? (total_scheduled + 1) / pipeline_stages : 0;
    balance_cost += std::max<int64_t>(0, stage_load[cycle] - ideal_load);
    balance_cost += stage_bit_load[cycle] / 128;
    if (cycle > 0) {
      balance_cost += std::max<int64_t>(
          0, stage_load[cycle] + 1 - stage_load[cycle - 1] - 1);
    }
    if (cycle + 1 < static_cast<int64_t>(stage_load.size())) {
      balance_cost += std::max<int64_t>(
          0, stage_load[cycle] + 1 - stage_load[cycle + 1] - 1);
    }
  }

  const int64_t early_bias =
      (cycle - lb) *
      std::max<int64_t>(1, (reverse + fanout + fanin) / (mobility + 1));
  const int64_t center = pipeline_stages > 0 ? (pipeline_stages - 1) / 2 : 0;
  const int64_t center_spread_bias =
      std::abs(cycle - center) * std::max<int64_t>(1, bits / 32);
  const int64_t chain_bonus =
      (cycle == latest_operand_cycle ? operand_same_stage_bits : 0);

  return operand_boundary_cost * 8192LL +
         user_boundary_cost * 4096LL +
         user_forcing_cost * 16384LL +
         balance_cost * 64LL +
         early_bias * 128LL +
         center_spread_bias -
         chain_bonus * 8LL;
}

}  // namespace

namespace {

int64_t AgentNodeDelayPsOrZero(const DelayEstimator& delay_estimator,
                              Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentTimedUserCount(Node* node) {
  int64_t count = 0;
  for (Node* user : node->users()) {
    if (!IsUntimed(user)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentTimedOperandCount(Node* node) {
  int64_t count = 0;
  for (Node* operand : node->operands()) {
    if (!IsUntimed(operand)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentReverseDepth(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& reverse_depth) {
  auto it = reverse_depth.find(node);
  return it == reverse_depth.end() ? 0 : it->second;
}

int64_t AgentStageTargetPrefix(int64_t total_timed_nodes, int64_t pipeline_stages,
                               int64_t cycle) {
  if (pipeline_stages <= 0) {
    return total_timed_nodes;
  }
  const int64_t numer = (cycle + 1) * total_timed_nodes;
  return (numer + pipeline_stages - 1) / pipeline_stages;
}

int64_t AgentCandidateScore(
    Node* node, int64_t cycle, int64_t lb, int64_t ub, int64_t pipeline_stages,
    int64_t timed_node_count, const DelayEstimator& delay_estimator,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& reverse_depth,
    const std::vector<int64_t>& stage_load,
    const std::vector<int64_t>& stage_bits, sched::ScheduleBounds* bounds) {
  const int64_t bits = NodeBitCount(node);
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t fanout = std::max<int64_t>(1, AgentTimedUserCount(node));
  const int64_t fanin = std::max<int64_t>(1, AgentTimedOperandCount(node));
  const int64_t rev_depth = AgentReverseDepth(node, reverse_depth);
  const int64_t node_delay_ps = AgentNodeDelayPsOrZero(delay_estimator, node);

  int64_t operand_reg_cost = 0;
  int64_t same_stage_operand_bits = 0;
  int64_t latest_operand_cycle = lb;
  int64_t same_stage_pred_delay = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    latest_operand_cycle = std::max(latest_operand_cycle, it->second);
    if (it->second < cycle) {
      operand_reg_cost += NodeBitCount(operand) * (cycle - it->second);
    } else if (it->second == cycle) {
      same_stage_operand_bits += NodeBitCount(operand);
      same_stage_pred_delay += AgentNodeDelayPsOrZero(delay_estimator, operand);
    }
  }

  int64_t user_liveout_cost = 0;
  int64_t user_push_cost = 0;
  int64_t user_same_stage_pull = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto ait = assigned_cycles.find(user);
    if (ait != assigned_cycles.end()) {
      if (cycle < ait->second) {
        user_liveout_cost += bits * (ait->second - cycle);
      }
      if (cycle == ait->second) {
        user_same_stage_pull += bits;
      }
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    const int64_t user_ub = std::min<int64_t>(bounds->ub(user), pipeline_stages - 1);
    const int64_t user_mobility = std::max<int64_t>(0, user_ub - user_lb);
    if (cycle < user_lb) {
      user_liveout_cost += bits * (user_lb - cycle);
    } else if (cycle > user_lb) {
      const int64_t delta = cycle - user_lb;
      user_push_cost +=
          delta * bits *
          (std::max<int64_t>(1, NodeFanout(user)) +
           std::max<int64_t>(1, NodeBitCount(user) / 16 + 1)) /
          std::max<int64_t>(1, user_mobility + 1);
    }
  }

  int64_t stage_balance_cost = 0;
  if (cycle >= 0 && cycle < static_cast<int64_t>(stage_load.size())) {
    const int64_t prefix_load_after = [&]() {
      int64_t v = 0;
      for (int64_t i = 0; i <= cycle; ++i) {
        v += stage_load[i];
      }
      return v + 1;
    }();
    const int64_t prefix_target =
        AgentStageTargetPrefix(timed_node_count, pipeline_stages, cycle);
    stage_balance_cost +=
        std::max<int64_t>(0, prefix_load_after - prefix_target);
    stage_balance_cost += stage_load[cycle];
    stage_balance_cost += stage_bits[cycle] / 128;
    if (cycle > 0) {
      stage_balance_cost +=
          std::max<int64_t>(0, stage_load[cycle] + 1 - stage_load[cycle - 1] - 2);
    }
    if (cycle + 1 < static_cast<int64_t>(stage_load.size())) {
      stage_balance_cost +=
          std::max<int64_t>(0, stage_load[cycle] + 1 - stage_load[cycle + 1] - 2);
    }
  }

  const int64_t early_bias =
      (cycle - lb) *
      std::max<int64_t>(1, (rev_depth + fanout + fanin) / (mobility + 1));
  const int64_t center = pipeline_stages > 0 ? (pipeline_stages - 1) / 2 : 0;
  const int64_t center_bias =
      std::abs(cycle - center) * std::max<int64_t>(1, bits / 64);
  const int64_t stage_use_bonus =
      (cycle == latest_operand_cycle ? same_stage_operand_bits : 0);
  const int64_t timing_penalty =
      std::max<int64_t>(0, same_stage_pred_delay + node_delay_ps - 25000);

  return operand_reg_cost * 8192LL +
         user_liveout_cost * 4096LL +
         user_push_cost * 16384LL +
         stage_balance_cost * 64LL +
         early_bias * 128LL +
         center_bias * 4LL +
         timing_penalty * timing_penalty -
         stage_use_bonus * 16LL -
         user_same_stage_pull * 8LL;
}

}  // namespace

namespace {

int64_t AgentTimedNodeCount(FunctionBase* f) {
  int64_t count = 0;
  for (Node* node : f->nodes()) {
    if (!IsUntimed(node)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentTimedUserCountOnly(Node* node) {
  int64_t count = 0;
  for (Node* user : node->users()) {
    if (!IsUntimed(user)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentTimedOperandCountOnly(Node* node) {
  int64_t count = 0;
  for (Node* operand : node->operands()) {
    if (!IsUntimed(operand)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentReverseDepthValue(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& reverse_depth) {
  auto it = reverse_depth.find(node);
  return it == reverse_depth.end() ? 0 : it->second;
}

int64_t AgentPrefixTarget(int64_t total_timed_nodes, int64_t pipeline_stages,
                          int64_t cycle) {
  if (pipeline_stages <= 0) {
    return total_timed_nodes;
  }
  return ((cycle + 1) * total_timed_nodes + pipeline_stages - 1) /
         pipeline_stages;
}

int64_t AgentSumPrefixLoad(const std::vector<int64_t>& stage_load,
                           int64_t cycle) {
  int64_t total = 0;
  for (int64_t i = 0; i <= cycle && i < static_cast<int64_t>(stage_load.size());
       ++i) {
    total += stage_load[i];
  }
  return total;
}

int64_t AgentCandidateBoundaryScore(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t pipeline_stages, int64_t timed_node_count,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& reverse_depth,
    const std::vector<int64_t>& stage_load,
    const std::vector<int64_t>& stage_bits, sched::ScheduleBounds* bounds) {
  const int64_t node_bits = NodeBitCount(node);
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t reverse = AgentReverseDepthValue(node, reverse_depth);
  const int64_t fanout = std::max<int64_t>(int64_t{1}, AgentTimedUserCountOnly(node));
  const int64_t fanin =
      std::max<int64_t>(int64_t{1}, AgentTimedOperandCountOnly(node));

  int64_t operand_boundary_cost = 0;
  int64_t same_stage_operand_bits = 0;
  int64_t latest_operand_cycle = lb;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    latest_operand_cycle = std::max(latest_operand_cycle, it->second);
    if (it->second < candidate_cycle) {
      operand_boundary_cost +=
          NodeBitCount(operand) * (candidate_cycle - it->second);
    } else if (it->second == candidate_cycle) {
      same_stage_operand_bits += NodeBitCount(operand);
    }
  }

  int64_t user_boundary_cost = 0;
  int64_t user_forced_later_cost = 0;
  int64_t urgent_user_pressure = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    if (it != assigned_cycles.end()) {
      if (candidate_cycle < it->second) {
        user_boundary_cost += node_bits * (it->second - candidate_cycle);
      }
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    const int64_t user_ub =
        std::min<int64_t>(bounds->ub(user), pipeline_stages - 1);
    const int64_t user_mobility = std::max<int64_t>(0, user_ub - user_lb);
    if (candidate_cycle < user_lb) {
      user_boundary_cost += node_bits * (user_lb - candidate_cycle);
    } else if (candidate_cycle > user_lb) {
      const int64_t delta = candidate_cycle - user_lb;
      const int64_t weight =
          std::max<int64_t>(1, NodeFanout(user)) +
          std::max<int64_t>(1, NodeBitCount(user) / 16);
      user_forced_later_cost +=
          delta * node_bits * weight / std::max<int64_t>(1, user_mobility + 1);
      urgent_user_pressure +=
          delta * std::max<int64_t>(1, NodeBitCount(user) / 8 + 1) /
          std::max<int64_t>(1, user_mobility + 1);
    }
  }

  int64_t balance_cost = 0;
  if (candidate_cycle >= 0 &&
      candidate_cycle < static_cast<int64_t>(stage_load.size())) {
    const int64_t prefix_after =
        AgentSumPrefixLoad(stage_load, candidate_cycle) + 1;
    const int64_t prefix_target =
        AgentPrefixTarget(timed_node_count, pipeline_stages, candidate_cycle);
    balance_cost += std::max<int64_t>(0, prefix_after - prefix_target);
    balance_cost += stage_load[candidate_cycle];
    balance_cost += stage_bits[candidate_cycle] / 128;
    if (candidate_cycle > 0) {
      balance_cost += std::max<int64_t>(
          0, stage_load[candidate_cycle] + 1 - stage_load[candidate_cycle - 1] - 2);
    }
    if (candidate_cycle + 1 < static_cast<int64_t>(stage_load.size())) {
      balance_cost += std::max<int64_t>(
          0, stage_load[candidate_cycle] + 1 - stage_load[candidate_cycle + 1] - 2);
    }
  }

  const int64_t center = pipeline_stages > 0 ? (pipeline_stages - 1) / 2 : 0;
  const int64_t center_bias =
      std::abs(candidate_cycle - center) * std::max<int64_t>(1, node_bits / 64);
  const int64_t lateness_bias =
      (candidate_cycle - lb) *
      std::max<int64_t>(1, (reverse + fanout + fanin) / (mobility + 1));
  const int64_t chain_bonus =
      candidate_cycle == latest_operand_cycle ? same_stage_operand_bits : 0;

  return operand_boundary_cost * 8192LL + user_boundary_cost * 4096LL +
         user_forced_later_cost * 16384LL + urgent_user_pressure * 512LL +
         balance_cost * 64LL + lateness_bias * 128LL + center_bias * 4LL -
         chain_bonus * 16LL;
}

}  // namespace

namespace {

int64_t AgentSchedDelay(Node* node, const DelayEstimator& delay_estimator) {
  absl::StatusOr<int64_t> d = delay_estimator.GetOperationDelayInPs(node);
  return d.ok() ? d.value() : 0;
}

int64_t AgentSchedTimedUserCountLocal(Node* node) {
  int64_t count = 0;
  for (Node* user : node->users()) {
    if (!IsUntimed(user)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentSchedTimedOperandCountLocal(Node* node) {
  int64_t count = 0;
  for (Node* operand : node->operands()) {
    if (!IsUntimed(operand)) {
      ++count;
    }
  }
  return count;
}

int64_t AgentSchedMaxOperandCycle(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t result = 0;
  bool any = false;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it != assigned_cycles.end()) {
      result = any ? std::max(result, it->second) : it->second;
      any = true;
    }
  }
  return any ? result : 0;
}

int64_t AgentSchedMinUserLb(Node* node, sched::ScheduleBounds* bounds,
                            int64_t default_value) {
  int64_t result = default_value;
  bool any = false;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    int64_t lb = bounds->lb(user);
    result = any ? std::min(result, lb) : lb;
    any = true;
  }
  return any ? result : default_value;
}

int64_t AgentSchedReverseDepthLocal(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& reverse_depth) {
  auto it = reverse_depth.find(node);
  return it == reverse_depth.end() ? 0 : it->second;
}

int64_t AgentSchedPrefixTarget(int64_t total_nodes, int64_t pipeline_stages,
                               int64_t cycle) {
  return ((cycle + 1) * total_nodes + pipeline_stages - 1) / pipeline_stages;
}

int64_t AgentSchedPrefixLoad(const std::vector<int64_t>& stage_load,
                             int64_t cycle) {
  int64_t sum = 0;
  for (int64_t i = 0; i <= cycle; ++i) {
    sum += stage_load[i];
  }
  return sum;
}

int64_t AgentSchedPriority(Node* node, sched::ScheduleBounds* bounds,
                           const absl::flat_hash_map<Node*, int64_t>& reverse_depth) {
  const int64_t lb = bounds->lb(node);
  const int64_t ub = bounds->ub(node);
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t bits = NodeBitCount(node);
  const int64_t users = std::max<int64_t>(1, AgentSchedTimedUserCountLocal(node));
  const int64_t fanin = std::max<int64_t>(1, AgentSchedTimedOperandCountLocal(node));
  const int64_t depth = AgentSchedReverseDepthLocal(node, reverse_depth);
  return mobility * 1000000000LL - depth * 1000000LL -
         bits * 1024LL - users * 64LL - fanin;
}

int64_t AgentSchedCandidateCost(
    Node* node, int64_t cycle, int64_t lb, int64_t ub, int64_t pipeline_stages,
    int64_t timed_node_count,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& reverse_depth,
    const std::vector<int64_t>& stage_load,
    const std::vector<int64_t>& stage_bits, sched::ScheduleBounds* bounds) {
  const int64_t bits = NodeBitCount(node);
  const int64_t users = std::max<int64_t>(1, AgentSchedTimedUserCountLocal(node));
  const int64_t depth = AgentSchedReverseDepthLocal(node, reverse_depth);
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t latest_operand_cycle =
      AgentSchedMaxOperandCycle(node, assigned_cycles);
  const int64_t earliest_user_lb =
      AgentSchedMinUserLb(node, bounds, ub);

  int64_t operand_boundary_cost = 0;
  int64_t operand_same_stage_bonus = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it == assigned_cycles.end()) {
      continue;
    }
    if (it->second < cycle) {
      operand_boundary_cost +=
          NodeBitCount(operand) * (cycle - it->second);
    } else if (it->second == cycle) {
      operand_same_stage_bonus += NodeBitCount(operand);
    }
  }

  int64_t liveout_cost = 0;
  int64_t push_user_cost = 0;
  int64_t assigned_user_pull_bonus = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    if (it != assigned_cycles.end()) {
      if (cycle < it->second) {
        liveout_cost += bits * (it->second - cycle);
      } else if (cycle == it->second) {
        assigned_user_pull_bonus += bits;
      }
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    const int64_t user_ub = std::min<int64_t>(bounds->ub(user), pipeline_stages - 1);
    const int64_t user_mobility = std::max<int64_t>(0, user_ub - user_lb);
    const int64_t user_bits = NodeBitCount(user);
    if (cycle < user_lb) {
      liveout_cost += bits * (user_lb - cycle);
    } else if (cycle > user_lb) {
      const int64_t delta = cycle - user_lb;
      push_user_cost +=
          delta * bits *
          (std::max<int64_t>(1, AgentSchedTimedUserCountLocal(user)) +
           std::max<int64_t>(1, user_bits / 16 + 1)) /
          std::max<int64_t>(1, user_mobility + 1);
    }
  }

  int64_t balance_cost = 0;
  const int64_t prefix_after = AgentSchedPrefixLoad(stage_load, cycle) + 1;
  const int64_t prefix_target =
      AgentSchedPrefixTarget(timed_node_count, pipeline_stages, cycle);
  balance_cost += std::max<int64_t>(0, prefix_after - prefix_target);
  balance_cost += stage_load[cycle];
  balance_cost += stage_bits[cycle] / 256;

  const int64_t lateness = cycle - lb;
  const int64_t center = (pipeline_stages - 1) / 2;
  const int64_t center_bias = std::abs(cycle - center);

  const bool prefer_late =
      mobility >= 2 && bits >= 32 && users >= 2;
  const int64_t critical_asap_bias =
      lateness * std::max<int64_t>(1, (depth + users) / (mobility + 1));
  const int64_t flexible_late_bias =
      std::max<int64_t>(0, earliest_user_lb - cycle) * std::max<int64_t>(1, bits / 32);

  int64_t cost = 0;
  cost += operand_boundary_cost * 8192LL;
  cost += liveout_cost * 4096LL;
  cost += push_user_cost * 16384LL;
  cost += balance_cost * 128LL;
  cost += center_bias * std::max<int64_t>(1, bits / 64);
  cost -= operand_same_stage_bonus * 16LL;
  cost -= assigned_user_pull_bonus * 8LL;

  if (prefer_late) {
    cost += flexible_late_bias * 32LL;
    cost += lateness * std::max<int64_t>(1, depth / 4);
  } else {
    cost += critical_asap_bias * 256LL;
  }

  if (cycle == latest_operand_cycle) {
    cost -= operand_same_stage_bonus * 8LL;
  }

  return cost;
}

}  // namespace

absl::StatusOr<ScheduleCycleMap> AgentGeneratedScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints) {
  XLS_RET_CHECK_GT(pipeline_stages, 0);
  (void)clock_period_ps;

  for (const SchedulingConstraint& constraint : constraints) {
    if (std::holds_alternative<RecvsFirstSendsLastConstraint>(constraint)) {
      for (Node* node : f->nodes()) {
        if (node->Is<Receive>()) {
          XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, 0));
        }
        if (node->Is<Send>()) {
          XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(node, pipeline_stages - 1));
        }
      }
    } else if (std::holds_alternative<NodeInCycleConstraint>(constraint)) {
      const NodeInCycleConstraint& nic =
          std::get<NodeInCycleConstraint>(constraint);
      XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(nic.GetNode(), nic.GetCycle()));
      XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(nic.GetNode(), nic.GetCycle()));
    }
  }

  XLS_RETURN_IF_ERROR(bounds->PropagateBounds());
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_nodes, TopoSort(f));

  absl::flat_hash_map<Node*, int64_t> reverse_depth;
  reverse_depth.reserve(topo_nodes.size());
  for (auto it = topo_nodes.rbegin(); it != topo_nodes.rend(); ++it) {
    Node* node = *it;
    if (IsUntimed(node)) {
      continue;
    }
    int64_t depth = 0;
    for (Node* user : node->users()) {
      if (IsUntimed(user)) {
        continue;
      }
      auto found = reverse_depth.find(user);
      depth = std::max(
          depth, (found == reverse_depth.end() ? int64_t{0} : found->second) + 1);
    }
    reverse_depth[node] = depth;
  }

  std::vector<Node*> timed_nodes;
  timed_nodes.reserve(topo_nodes.size());
  for (Node* node : topo_nodes) {
    if (!IsUntimed(node)) {
      timed_nodes.push_back(node);
    }
  }

  std::stable_sort(timed_nodes.begin(), timed_nodes.end(),
                   [&](Node* a, Node* b) {
                     const int64_t pa = AgentSchedPriority(a, bounds, reverse_depth);
                     const int64_t pb = AgentSchedPriority(b, bounds, reverse_depth);
                     if (pa != pb) {
                       return pa < pb;
                     }
                     return a->id() < b->id();
                   });

  const int64_t timed_node_count = static_cast<int64_t>(timed_nodes.size());

  ScheduleCycleMap cycle_map;
  absl::flat_hash_map<Node*, int64_t> assigned_cycles;
  cycle_map.reserve(timed_node_count);
  assigned_cycles.reserve(timed_node_count);

  std::vector<int64_t> stage_load(pipeline_stages, 0);
  std::vector<int64_t> stage_bits(pipeline_stages, 0);

  for (Node* node : timed_nodes) {
    int64_t lb = bounds->lb(node);
    for (Node* operand : node->operands()) {
      if (IsUntimed(operand)) {
        continue;
      }
      auto it = assigned_cycles.find(operand);
      if (it != assigned_cycles.end()) {
        lb = std::max(lb, it->second);
      }
    }

    int64_t ub = std::min<int64_t>(bounds->ub(node), pipeline_stages - 1);
    lb = std::min(lb, ub);

    int64_t best_cycle = lb;
    int64_t best_score = std::numeric_limits<int64_t>::max();

    for (int64_t cycle = lb; cycle <= ub; ++cycle) {
      int64_t score = AgentSchedCandidateCost(
          node, cycle, lb, ub, pipeline_stages, timed_node_count,
          assigned_cycles, reverse_depth, stage_load, stage_bits, bounds);
      score += AgentSchedDelay(node, delay_estimator) / 1024;
      if (score < best_score || (score == best_score && cycle < best_cycle)) {
        best_score = score;
        best_cycle = cycle;
      }
    }

    assigned_cycles[node] = best_cycle;
    cycle_map[node] = best_cycle;
    stage_load[best_cycle] += 1;
    stage_bits[best_cycle] += NodeBitCount(node);

    XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(node, best_cycle));
    XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, best_cycle));
  }

  return cycle_map;
}























}  // namespace xls

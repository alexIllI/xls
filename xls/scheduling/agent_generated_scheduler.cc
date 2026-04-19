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

int64_t EstimateBoundaryRegisterCost(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t cost = 0;
  for (Node* operand : node->operands()) {
    auto it = assigned_cycles.find(operand);
    if (it != assigned_cycles.end() && it->second < candidate_cycle) {
      cost += NodeBitCount(operand);
    }
  }
  for (Node* user : node->users()) {
    auto it = assigned_cycles.find(user);
    if (it != assigned_cycles.end() && candidate_cycle < it->second) {
      cost += NodeBitCount(node);
    }
  }
  return cost;
}

int64_t ScoreCandidateCycle(
    Node* node, int64_t candidate_cycle, int64_t earliest_cycle,
    int64_t latest_cycle, int64_t clock_period_ps, int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count) {
  int64_t start_time_ps = 0;
  for (Node* operand : node->operands()) {
    auto cycle_it = assigned_cycles.find(operand);
    if (cycle_it != assigned_cycles.end() &&
        cycle_it->second == candidate_cycle) {
      auto time_it = completion_time_ps.find(operand);
      if (time_it != completion_time_ps.end()) {
        start_time_ps = std::max(start_time_ps, time_it->second);
      }
    }
  }

  const int64_t completion_ps = start_time_ps + node_delay_ps;
  const int64_t timing_overflow_ps =
      std::max<int64_t>(0, completion_ps - clock_period_ps);
  const int64_t mobility_span =
      std::max<int64_t>(1, latest_cycle - earliest_cycle + 1);
  const int64_t stage_fill = stage_node_count.contains(candidate_cycle)
                                 ? stage_node_count.at(candidate_cycle)
                                 : 0;
  const int64_t boundary_cost =
      EstimateBoundaryRegisterCost(node, candidate_cycle, assigned_cycles);
  const int64_t lateness_cost = candidate_cycle - earliest_cycle;
  const int64_t criticality_cost =
      NodeBitCount(node) * NodeFanout(node) * lateness_cost;

  return timing_overflow_ps * timing_overflow_ps * 1000000LL +
         boundary_cost * 64 + stage_fill * 32 +
         (criticality_cost * 16) / mobility_span + lateness_cost;
}

}  // namespace

namespace {

int64_t GetNodeDelayPsOrZero(const DelayEstimator& delay_estimator, Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

bool IsTimedAssigned(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  return !IsUntimed(node) && assigned_cycles.contains(node);
}

int64_t AssignedCycleOrSelfLb(
    Node* node, const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    sched::ScheduleBounds* bounds) {
  if (IsUntimed(node)) {
    return 0;
  }
  auto it = assigned_cycles.find(node);
  if (it != assigned_cycles.end()) {
    return it->second;
  }
  return bounds->lb(node);
}

int64_t EarliestSameStageStartTime(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps) {
  int64_t start_time_ps = 0;
  for (Node* operand : node->operands()) {
    auto cycle_it = assigned_cycles.find(operand);
    if (cycle_it != assigned_cycles.end() &&
        cycle_it->second == candidate_cycle) {
      auto time_it = completion_time_ps.find(operand);
      if (time_it != completion_time_ps.end()) {
        start_time_ps = std::max(start_time_ps, time_it->second);
      }
    }
  }
  return start_time_ps;
}

int64_t EstimateFutureUsePressure(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    sched::ScheduleBounds* bounds) {
  int64_t pressure = 0;
  const int64_t node_bits = NodeBitCount(node);
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    if (it != assigned_cycles.end()) {
      if (it->second > candidate_cycle) {
        pressure += node_bits * (it->second - candidate_cycle);
      }
      continue;
    }
    const int64_t projected_user_cycle = std::max(bounds->lb(user), candidate_cycle);
    if (projected_user_cycle > candidate_cycle) {
      pressure += node_bits * (projected_user_cycle - candidate_cycle);
    }
  }
  return pressure;
}

int64_t EstimateOperandBoundaryCost(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t cost = 0;
  for (Node* operand : node->operands()) {
    auto it = assigned_cycles.find(operand);
    if (it != assigned_cycles.end() && it->second < candidate_cycle) {
      cost += NodeBitCount(operand) * (candidate_cycle - it->second);
    }
  }
  return cost;
}

int64_t EstimateConsumerPullCost(
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

int64_t ScorePlacement(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t clock_period_ps, int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count,
    sched::ScheduleBounds* bounds) {
  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t same_stage_start = EarliestSameStageStartTime(
      node, candidate_cycle, assigned_cycles, completion_time_ps);
  const int64_t same_stage_end = same_stage_start + node_delay_ps;
  const int64_t timing_overflow =
      std::max<int64_t>(0, same_stage_end - clock_period_ps);

  const int64_t operand_boundary_cost =
      EstimateOperandBoundaryCost(node, candidate_cycle, assigned_cycles);
  const int64_t consumer_pull_cost =
      EstimateConsumerPullCost(node, candidate_cycle, assigned_cycles, bounds);
  const int64_t future_use_pressure =
      EstimateFutureUsePressure(node, candidate_cycle, assigned_cycles, bounds);

  const int64_t stage_fill =
      stage_node_count.contains(candidate_cycle)
          ? stage_node_count.at(candidate_cycle)
          : 0;

  int64_t predecessor_slack_penalty = 0;
  for (Node* operand : node->operands()) {
    if (IsUntimed(operand)) {
      continue;
    }
    auto it = assigned_cycles.find(operand);
    if (it != assigned_cycles.end() && it->second > candidate_cycle) {
      predecessor_slack_penalty += 1000000000LL;
    }
  }

  const int64_t late_bias = candidate_cycle - lb;
  const int64_t center_bias = std::abs(candidate_cycle - ((lb + ub) / 2));

  return timing_overflow * timing_overflow * 1000000000LL +
         predecessor_slack_penalty +
         operand_boundary_cost * 1024 +
         consumer_pull_cost * 2048 +
         future_use_pressure * 64 +
         stage_fill * 32 +
         late_bias * (mobility == 0 ? 256 : 8) +
         center_bias;
}

}  // namespace

namespace {

int64_t AgentAbsDiff(int64_t a, int64_t b) {
  return a > b ? a - b : b - a;
}

int64_t AgentNodeDelayPsOrZero(const DelayEstimator& delay_estimator,
                               Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentSameStageStartTime(
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

int64_t AgentOperandRegisterCost(
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

int64_t AgentProjectedValueLifetimeCost(
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
      if (it->second > candidate_cycle) {
        cost += node_bits * (it->second - candidate_cycle);
      }
      continue;
    }
    const int64_t projected_user_cycle = std::max(bounds->lb(user), candidate_cycle);
    if (projected_user_cycle > candidate_cycle) {
      cost += node_bits * (projected_user_cycle - candidate_cycle);
    }
  }
  return cost;
}

int64_t AgentForcedUserDelayCost(Node* node, int64_t candidate_cycle,
                                 sched::ScheduleBounds* bounds) {
  int64_t cost = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    if (candidate_cycle > user_lb) {
      const int64_t forced_late = candidate_cycle - user_lb;
      cost += forced_late *
              (NodeBitCount(node) + NodeBitCount(user) * NodeFanout(user));
    }
  }
  return cost;
}

int64_t AgentStageFillCost(
    int64_t cycle,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count) {
  auto it = stage_node_count.find(cycle);
  return it == stage_node_count.end() ? 0 : it->second;
}

int64_t AgentScoreCycle(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t clock_period_ps, int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count,
    sched::ScheduleBounds* bounds) {
  const int64_t same_stage_start =
      AgentSameStageStartTime(node, candidate_cycle, assigned_cycles,
                              completion_time_ps);
  const int64_t same_stage_end = same_stage_start + node_delay_ps;
  const int64_t timing_overflow =
      std::max<int64_t>(0, same_stage_end - clock_period_ps);

  const int64_t operand_cost =
      AgentOperandRegisterCost(node, candidate_cycle, assigned_cycles);
  const int64_t value_lifetime_cost =
      AgentProjectedValueLifetimeCost(node, candidate_cycle, assigned_cycles,
                                      bounds);
  const int64_t forced_user_delay_cost =
      AgentForcedUserDelayCost(node, candidate_cycle, bounds);
  const int64_t stage_fill_cost =
      AgentStageFillCost(candidate_cycle, stage_node_count);

  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t late_bias = candidate_cycle - lb;
  const int64_t center_bias = AgentAbsDiff(candidate_cycle, (lb + ub) / 2);
  const int64_t fanout = NodeFanout(node);
  const int64_t critical_bias =
      (late_bias * std::max<int64_t>(1, NodeBitCount(node) / 8) * fanout) /
      std::max<int64_t>(1, mobility + 1);

  return timing_overflow * timing_overflow * 1000000000LL +
         operand_cost * 4096LL +
         value_lifetime_cost * 2048LL +
         forced_user_delay_cost * 8192LL +
         stage_fill_cost * 64LL +
         critical_bias * 32LL +
         center_bias;
}

}  // namespace

namespace {

int64_t AgentSchedAbs(int64_t x) { return x < 0 ? -x : x; }

int64_t AgentSchedDelayPsOrZero(const DelayEstimator& delay_estimator,
                                Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentSchedSameCycleStartTime(
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

int64_t AgentSchedOperandBoundaryCost(
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

int64_t AgentSchedAssignedUserCost(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles) {
  int64_t cost = 0;
  const int64_t node_bits = NodeBitCount(node);
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    if (it == assigned_cycles.end()) {
      continue;
    }
    if (candidate_cycle < it->second) {
      cost += node_bits * (it->second - candidate_cycle);
    }
  }
  return cost;
}

int64_t AgentSchedForcedLaterUserCost(Node* node, int64_t candidate_cycle,
                                      sched::ScheduleBounds* bounds) {
  int64_t cost = 0;
  const int64_t node_bits = NodeBitCount(node);
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    const int64_t user_lb = bounds->lb(user);
    if (candidate_cycle > user_lb) {
      const int64_t delta = candidate_cycle - user_lb;
      const int64_t mobility = std::max<int64_t>(0, bounds->ub(user) - user_lb);
      const int64_t urgency = std::max<int64_t>(1, NodeFanout(user));
      cost += delta * node_bits * urgency / std::max<int64_t>(1, mobility + 1);
      cost += delta * std::max<int64_t>(1, NodeBitCount(user) / 8);
    }
  }
  return cost;
}

int64_t AgentSchedStageDensityCost(
    int64_t candidate_cycle,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count) {
  auto it = stage_node_count.find(candidate_cycle);
  return it == stage_node_count.end() ? 0 : it->second;
}

int64_t AgentSchedCriticalityBias(Node* node, int64_t candidate_cycle,
                                  int64_t lb, int64_t ub) {
  const int64_t slack = std::max<int64_t>(0, ub - lb);
  const int64_t lateness = candidate_cycle - lb;
  const int64_t midpoint = (lb + ub) / 2;
  const int64_t center_dist = AgentSchedAbs(candidate_cycle - midpoint);
  const int64_t urgency =
      std::max<int64_t>(1, NodeFanout(node) + NodeBitCount(node) / 16);
  return (lateness * urgency * 8) / std::max<int64_t>(1, slack + 1) + center_dist;
}

int64_t AgentSchedScoreCandidate(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t clock_period_ps, int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count,
    sched::ScheduleBounds* bounds) {
  const int64_t start_time_ps = AgentSchedSameCycleStartTime(
      node, candidate_cycle, assigned_cycles, completion_time_ps);
  const int64_t end_time_ps = start_time_ps + node_delay_ps;
  const int64_t timing_overflow_ps =
      std::max<int64_t>(0, end_time_ps - clock_period_ps);

  const int64_t operand_boundary_cost =
      AgentSchedOperandBoundaryCost(node, candidate_cycle, assigned_cycles);
  const int64_t assigned_user_cost =
      AgentSchedAssignedUserCost(node, candidate_cycle, assigned_cycles);
  const int64_t forced_later_user_cost =
      AgentSchedForcedLaterUserCost(node, candidate_cycle, bounds);
  const int64_t stage_density_cost =
      AgentSchedStageDensityCost(candidate_cycle, stage_node_count);
  const int64_t criticality_bias =
      AgentSchedCriticalityBias(node, candidate_cycle, lb, ub);

  return timing_overflow_ps * timing_overflow_ps * 1000000000LL +
         operand_boundary_cost * 4096LL +
         assigned_user_cost * 2048LL +
         forced_later_user_cost * 8192LL +
         stage_density_cost * 32LL + criticality_bias;
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

int64_t AgentMobilityAbs(int64_t x) { return x < 0 ? -x : x; }

int64_t AgentMobilityDelayPsOrZero(const DelayEstimator& delay_estimator,
                                   Node* node) {
  absl::StatusOr<int64_t> delay_or = delay_estimator.GetOperationDelayInPs(node);
  return delay_or.ok() ? delay_or.value() : 0;
}

int64_t AgentMobilitySameStageStartTime(
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

int64_t AgentMobilityOperandCarryCost(
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

int64_t AgentMobilityProjectedUseCost(
    Node* node, int64_t candidate_cycle,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const sched::ScheduleBounds& trial_bounds) {
  int64_t cost = 0;
  const int64_t node_bits = NodeBitCount(node);
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    auto it = assigned_cycles.find(user);
    const int64_t user_cycle =
        it != assigned_cycles.end() ? it->second : trial_bounds.lb(user);
    if (user_cycle > candidate_cycle) {
      cost += node_bits * (user_cycle - candidate_cycle);
    }
  }
  return cost;
}

int64_t AgentMobilityForcedSuccessorCost(Node* node, int64_t candidate_cycle,
                                         const sched::ScheduleBounds& bounds) {
  int64_t cost = 0;
  const int64_t node_bits = NodeBitCount(node);
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    const int64_t user_lb = bounds.lb(user);
    const int64_t user_mobility =
        std::max<int64_t>(0, bounds.ub(user) - user_lb);
    if (candidate_cycle > user_lb) {
      const int64_t delta = candidate_cycle - user_lb;
      cost += delta * (node_bits + NodeBitCount(user) * NodeFanout(user)) /
              std::max<int64_t>(1, user_mobility + 1);
    }
  }
  return cost;
}

int64_t AgentMobilitySuccessorUrgency(Node* node,
                                      const sched::ScheduleBounds& bounds) {
  int64_t urgency = 0;
  for (Node* user : node->users()) {
    if (IsUntimed(user)) {
      continue;
    }
    const int64_t user_lb = bounds.lb(user);
    const int64_t user_mobility =
        std::max<int64_t>(0, bounds.ub(user) - user_lb);
    urgency +=
        (std::max<int64_t>(1, NodeFanout(user)) +
         std::max<int64_t>(1, NodeBitCount(user) / 16)) *
        64 / std::max<int64_t>(1, user_mobility + 1);
  }
  return urgency;
}

bool AgentMobilityReadyNodeLess(Node* lhs, Node* rhs,
                                const sched::ScheduleBounds& bounds,
                                const DelayEstimator& delay_estimator) {
  const int64_t lhs_lb = bounds.lb(lhs);
  const int64_t rhs_lb = bounds.lb(rhs);
  const int64_t lhs_mobility =
      std::max<int64_t>(0, bounds.ub(lhs) - lhs_lb);
  const int64_t rhs_mobility =
      std::max<int64_t>(0, bounds.ub(rhs) - rhs_lb);
  if (lhs_mobility != rhs_mobility) {
    return lhs_mobility < rhs_mobility;
  }

  const int64_t lhs_urgency = AgentMobilitySuccessorUrgency(lhs, bounds);
  const int64_t rhs_urgency = AgentMobilitySuccessorUrgency(rhs, bounds);
  if (lhs_urgency != rhs_urgency) {
    return lhs_urgency > rhs_urgency;
  }

  if (lhs_lb != rhs_lb) {
    return lhs_lb < rhs_lb;
  }

  const int64_t lhs_delay = AgentMobilityDelayPsOrZero(delay_estimator, lhs);
  const int64_t rhs_delay = AgentMobilityDelayPsOrZero(delay_estimator, rhs);
  if (lhs_delay != rhs_delay) {
    return lhs_delay > rhs_delay;
  }

  const int64_t lhs_width = NodeBitCount(lhs) * NodeFanout(lhs);
  const int64_t rhs_width = NodeBitCount(rhs) * NodeFanout(rhs);
  if (lhs_width != rhs_width) {
    return lhs_width > rhs_width;
  }

  return lhs->id() < rhs->id();
}

int64_t AgentMobilityScoreCandidate(
    Node* node, int64_t candidate_cycle, int64_t lb, int64_t ub,
    int64_t clock_period_ps, int64_t node_delay_ps,
    const absl::flat_hash_map<Node*, int64_t>& assigned_cycles,
    const absl::flat_hash_map<Node*, int64_t>& completion_time_ps,
    const absl::flat_hash_map<int64_t, int64_t>& stage_node_count,
    const sched::ScheduleBounds& current_bounds,
    const sched::ScheduleBounds& trial_bounds) {
  const int64_t start_time_ps = AgentMobilitySameStageStartTime(
      node, candidate_cycle, assigned_cycles, completion_time_ps);
  const int64_t end_time_ps = start_time_ps + node_delay_ps;
  const int64_t timing_overflow_ps =
      std::max<int64_t>(0, end_time_ps - clock_period_ps);

  const int64_t mobility = std::max<int64_t>(0, ub - lb);
  const int64_t width_pressure = NodeBitCount(node) * NodeFanout(node);
  const int64_t operand_carry_cost =
      AgentMobilityOperandCarryCost(node, candidate_cycle, assigned_cycles);
  const int64_t projected_use_cost =
      AgentMobilityProjectedUseCost(node, candidate_cycle, assigned_cycles,
                                    trial_bounds);
  const int64_t forced_successor_cost =
      AgentMobilityForcedSuccessorCost(node, candidate_cycle, current_bounds);

  int64_t preferred_cycle = lb;
  if (mobility > 0) {
    preferred_cycle =
        lb + (mobility * width_pressure) / (width_pressure + 64);
  }
  const int64_t target_distance =
      AgentMobilityAbs(candidate_cycle - preferred_cycle);

  const int64_t stage_load =
      stage_node_count.contains(candidate_cycle)
          ? stage_node_count.at(candidate_cycle)
          : 0;
  const int64_t lateness = candidate_cycle - lb;
  const int64_t critical_late_penalty =
      lateness * (256 / std::max<int64_t>(1, mobility + 1));

  return timing_overflow_ps * timing_overflow_ps * 1000000000LL +
         forced_successor_cost * 16384LL +
         projected_use_cost * 4096LL +
         operand_carry_cost * 2048LL +
         critical_late_penalty * 128LL +
         target_distance * 32LL + stage_load * 16LL;
}

}  // namespace

absl::StatusOr<ScheduleCycleMap> AgentGeneratedScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints) {
  if (std::getenv("XLS_AGENT_DRY_RUN") != nullptr) {
    XLS_RETURN_IF_ERROR(bounds->PropagateBounds());
    ScheduleCycleMap stub_map;
    for (Node* node : TopoSort(f)) {
      if (!IsUntimed(node)) {
        stub_map[node] = bounds->lb(node);
      }
    }
    return stub_map;
  }

  (void)pipeline_stages;
  (void)constraints;

  XLS_RETURN_IF_ERROR(bounds->PropagateBounds());

  auto topo = TopoSort(f);

  ScheduleCycleMap cycle_map;
  absl::flat_hash_map<Node*, int64_t> assigned_cycles;
  absl::flat_hash_map<Node*, int64_t> completion_time_ps;
  absl::flat_hash_map<int64_t, int64_t> stage_node_count;
  absl::flat_hash_map<Node*, int64_t> remaining_timed_operands;

  int64_t unscheduled_count = 0;
  for (Node* node : topo) {
    if (IsUntimed(node)) {
      continue;
    }
    int64_t timed_operands = 0;
    for (Node* operand : node->operands()) {
      if (!IsUntimed(operand)) {
        ++timed_operands;
      }
    }
    remaining_timed_operands[node] = timed_operands;
    ++unscheduled_count;
  }

  while (unscheduled_count > 0) {
    Node* selected_node = nullptr;
    for (Node* node : topo) {
      if (IsUntimed(node) || assigned_cycles.contains(node) ||
          remaining_timed_operands.at(node) != 0) {
        continue;
      }
      if (selected_node == nullptr ||
          AgentMobilityReadyNodeLess(node, selected_node, *bounds,
                                     delay_estimator)) {
        selected_node = node;
      }
    }

    if (selected_node == nullptr) {
      return absl::InternalError(
          "AgentGeneratedScheduler could not find a ready node.");
    }

    const int64_t lb = bounds->lb(selected_node);
    const int64_t ub = bounds->ub(selected_node);
    const int64_t node_delay_ps =
        AgentMobilityDelayPsOrZero(delay_estimator, selected_node);

    int64_t best_cycle = lb;
    int64_t best_score = std::numeric_limits<int64_t>::max();
    bool found_feasible_cycle = false;

    for (int64_t candidate_cycle = lb; candidate_cycle <= ub; ++candidate_cycle) {
      sched::ScheduleBounds trial_bounds = *bounds;
      if (!trial_bounds.TightenNodeLb(selected_node, candidate_cycle).ok()) {
        continue;
      }
      if (!trial_bounds.TightenNodeUb(selected_node, candidate_cycle).ok()) {
        continue;
      }
      if (!trial_bounds.PropagateBounds().ok()) {
        continue;
      }

      const int64_t score = AgentMobilityScoreCandidate(
          selected_node, candidate_cycle, lb, ub, clock_period_ps, node_delay_ps,
          assigned_cycles, completion_time_ps, stage_node_count, *bounds,
          trial_bounds);

      if (!found_feasible_cycle || score < best_score ||
          (score == best_score &&
           AgentMobilityAbs(candidate_cycle - lb) <
               AgentMobilityAbs(best_cycle - lb)) ||
          (score == best_score &&
           AgentMobilityAbs(candidate_cycle - lb) ==
               AgentMobilityAbs(best_cycle - lb) &&
           candidate_cycle < best_cycle)) {
        found_feasible_cycle = true;
        best_score = score;
        best_cycle = candidate_cycle;
      }
    }

    if (!found_feasible_cycle) {
      return absl::InternalError(
          "AgentGeneratedScheduler found no feasible cycle for a ready node.");
    }

    cycle_map[selected_node] = best_cycle;
    assigned_cycles[selected_node] = best_cycle;
    stage_node_count[best_cycle] += 1;
    completion_time_ps[selected_node] =
        AgentMobilitySameStageStartTime(selected_node, best_cycle, assigned_cycles,
                                        completion_time_ps) +
        node_delay_ps;

    XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(selected_node, best_cycle));
    XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(selected_node, best_cycle));
    XLS_RETURN_IF_ERROR(bounds->PropagateBounds());

    for (Node* user : selected_node->users()) {
      if (IsUntimed(user)) {
        continue;
      }
      auto it = remaining_timed_operands.find(user);
      if (it != remaining_timed_operands.end() && it->second > 0) {
        --it->second;
      }
    }

    --unscheduled_count;
  }

  return cycle_map;
}











}  // namespace xls

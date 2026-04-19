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

absl::StatusOr<ScheduleCycleMap> AgentGeneratedScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints) {
  (void)constraints;

  if (std::getenv("XLS_AGENT_DRY_RUN") != nullptr) {
    XLS_RETURN_IF_ERROR(bounds->PropagateBounds());
    ScheduleCycleMap stub_map;
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> stub_nodes, TopoSort(f));
    for (Node* node : stub_nodes) {
      if (!IsUntimed(node)) {
        stub_map[node] = bounds->lb(node);
      }
    }
    return stub_map;
  }

  // Call PropagateBounds ONCE before the loop — O(n).
  // Do NOT call it inside the loop (that would be O(n²) on large designs).
  XLS_RETURN_IF_ERROR(bounds->PropagateBounds());

  ScheduleCycleMap cycle_map;
  absl::flat_hash_map<Node*, int64_t> assigned_cycles;
  absl::flat_hash_map<Node*, int64_t> completion_time_ps;
  absl::flat_hash_map<int64_t, int64_t> stage_node_count;

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_nodes, TopoSort(f));
  for (Node* node : topo_nodes) {
    if (IsUntimed(node)) {
      continue;
    }

    // Compute lb manually: max of bounds->lb and all timed predecessor cycles.
    // This is O(degree) and replaces calling PropagateBounds() per node.
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

    // Cap ub to pipeline_stages - 1 when a fixed stage count is required.
    int64_t ub = bounds->ub(node);
    if (pipeline_stages > 0) {
      ub = std::min(ub, pipeline_stages - 1);
    }

    // Safety clamp: lb must not exceed ub.
    lb = std::min(lb, ub);

    const int64_t node_delay_ps =
        AgentListNodeDelayPsOrZero(delay_estimator, node);

    int64_t best_cycle = lb;
    int64_t best_score = std::numeric_limits<int64_t>::max();

    for (int64_t candidate_cycle = lb; candidate_cycle <= ub; ++candidate_cycle) {
      bool legal = true;
      for (Node* operand : node->operands()) {
        if (IsUntimed(operand)) {
          continue;
        }
        auto it = assigned_cycles.find(operand);
        if (it != assigned_cycles.end() && it->second > candidate_cycle) {
          legal = false;
          break;
        }
      }
      if (!legal) {
        continue;
      }

      const int64_t score = AgentListScoreCandidate(
          node, candidate_cycle, lb, ub, clock_period_ps, node_delay_ps,
          assigned_cycles, completion_time_ps, stage_node_count, bounds);

      if (score < best_score ||
          (score == best_score &&
           AgentListAbs(candidate_cycle - ((lb + ub) / 2)) <
               AgentListAbs(best_cycle - ((lb + ub) / 2))) ||
          (score == best_score &&
           AgentListAbs(candidate_cycle - ((lb + ub) / 2)) ==
               AgentListAbs(best_cycle - ((lb + ub) / 2)) &&
           candidate_cycle < best_cycle)) {
        best_score = score;
        best_cycle = candidate_cycle;
      }
    }

    cycle_map[node] = best_cycle;
    assigned_cycles[node] = best_cycle;
    stage_node_count[best_cycle] += 1;
    completion_time_ps[node] =
        AgentListSameStageStartTime(node, best_cycle, assigned_cycles,
                                    completion_time_ps) +
        node_delay_ps;

    // Tighten bounds for XLS verifier — O(1) each, no propagation needed.
    XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(node, best_cycle));
    XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, best_cycle));
  }

  return cycle_map;
}

}  // namespace xls

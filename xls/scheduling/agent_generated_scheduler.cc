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

absl::StatusOr<ScheduleCycleMap> AgentGeneratedScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints) {
  // `pipeline_stages` and `constraints` are embedded in `bounds` by the caller
  // (via RunPipelineSchedule). We cooperate with the bounds object rather than
  // re-deriving constraints here.
  (void)pipeline_stages;
  (void)constraints;

  // Propagate the initial bounds once so lb()/ub() reflect caller constraints.
  XLS_RETURN_IF_ERROR(bounds->PropagateBounds());

  ScheduleCycleMap cycle_map;
  absl::flat_hash_map<Node*, int64_t> assigned_cycles;
  absl::flat_hash_map<Node*, int64_t> completion_time_ps;
  absl::flat_hash_map<int64_t, int64_t> stage_node_count;

  for (Node* node : TopoSort(f)) {
    if (IsUntimed(node)) {
      continue;
    }

    const int64_t lb = bounds->lb(node);
    const int64_t ub = bounds->ub(node);

    // Estimate combinational delay for this node. If the delay model cannot
    // provide an estimate, treat the node as delay-free so we still emit a
    // valid schedule.
    int64_t node_delay_ps = 0;
    absl::StatusOr<int64_t> node_delay_or =
        delay_estimator.GetOperationDelayInPs(node);
    if (node_delay_or.ok()) {
      node_delay_ps = node_delay_or.value();
    }

    // Pick the best cycle in [lb, ub] using the scoring heuristic.
    int64_t best_cycle = lb;
    int64_t best_score = std::numeric_limits<int64_t>::max();
    for (int64_t candidate = lb; candidate <= ub; ++candidate) {
      const int64_t score = ScoreCandidateCycle(
          node, candidate, lb, ub, clock_period_ps, node_delay_ps,
          assigned_cycles, completion_time_ps, stage_node_count);
      if (score < best_score) {
        best_score = score;
        best_cycle = candidate;
      }
    }

    // Record the choice.
    cycle_map[node] = best_cycle;
    assigned_cycles[node] = best_cycle;
    stage_node_count[best_cycle] += 1;

    // Completion time within the chosen stage: start after same-stage
    // operands finish, then add this node's own delay.
    int64_t start_time_ps = 0;
    for (Node* operand : node->operands()) {
      auto cycle_it = assigned_cycles.find(operand);
      if (cycle_it != assigned_cycles.end() &&
          cycle_it->second == best_cycle) {
        auto time_it = completion_time_ps.find(operand);
        if (time_it != completion_time_ps.end()) {
          start_time_ps = std::max(start_time_ps, time_it->second);
        }
      }
    }
    completion_time_ps[node] = start_time_ps + node_delay_ps;

    // Pin the chosen cycle in the bounds and re-propagate so later nodes see
    // the tightened windows.
    if (lb != ub) {
      XLS_RETURN_IF_ERROR(bounds->TightenNodeLb(node, best_cycle));
      XLS_RETURN_IF_ERROR(bounds->TightenNodeUb(node, best_cycle));
      XLS_RETURN_IF_ERROR(bounds->PropagateBounds());
    }
  }

  return cycle_map;
}

}  // namespace xls

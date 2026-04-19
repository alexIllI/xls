# XLS — AlphaEvolve Fork

This is a fork of [google/xls](https://github.com/google/xls) maintained for use with [AlphaEvolve-XLS](https://github.com/alexIllI/alphaevolve-xls), an AI-driven evolutionary loop that automatically discovers better pipeline scheduling algorithms inside the XLS HLS toolchain.

---

## What this fork adds

The only purpose of this fork is to introduce a new scheduling strategy — `--scheduling_strategy=agent` — that dispatches to an AI-evolvable C++ function. Everything else in XLS is untouched.

### New files

| File | Purpose |
|------|---------|
| `xls/scheduling/agent_generated_scheduler.cc` | Standalone implementation of `AgentGeneratedScheduler()`. This is the function AlphaEvolve-XLS evolves — it is spliced, recompiled, and evaluated on every iteration. |
| `xls/scheduling/agent_generated_scheduler.h` | Header declaring `AgentGeneratedScheduler()`. |

### Modified files

| File | Change |
|------|--------|
| `xls/scheduling/BUILD` | Added `agent_generated_scheduler` library target and its dependencies. |
| `xls/scheduling/run_pipeline_schedule.cc` | Added `AGENT` dispatch branch — calls `AgentGeneratedScheduler()` when `strategy == AGENT`. |
| `xls/scheduling/scheduling_options.cc` | Added `AGENT` to the strategy string-to-enum mapping. |
| `xls/scheduling/scheduling_options.h` | Added `AGENT` to the `SchedulingStrategy` enum. |
| `xls/tools/scheduling_options_flags.cc` | Added `"agent"` as a valid value for `--scheduling_strategy`. |
| `xls/tools/scheduling_options_flags.proto` | Added `AGENT = 6` to the `SchedulingStrategyProto` enum. |

---

## How to use

Clone this fork and build the required targets:

```bash
git clone https://github.com/alexIllI/xls.git
cd xls

# Minimum build (--ppa_mode fast)
bazel build -c opt \
  //xls/scheduling:agent_generated_scheduler \
  //xls/tools:codegen_main \
  //xls/tools:opt_main \
  //xls/dslx/ir_convert:ir_converter_main

# Add this for --ppa_mode slow (ASAP7 area metrics, proc designs)
bazel build -c opt //xls/dev_tools:benchmark_main
```

Then follow the setup instructions in the [AlphaEvolve-XLS repository](https://github.com/alexIllI/alphaevolve-xls).

---

## Original project

Full documentation, tutorials, and the upstream codebase are at [google/xls](https://github.com/google/xls).
This fork makes no changes to any XLS scheduler, IR semantics, or toolchain behaviour outside the `agent` scheduling strategy path.

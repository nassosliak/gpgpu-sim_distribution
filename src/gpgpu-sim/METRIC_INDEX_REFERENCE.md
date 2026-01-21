# Performance Counter & Power Component Index Reference

This document provides the mapping of feature names to their access indices in the ML classifier integration.

## Performance Counter Indices (perf_count_t)

From `/accel-sim-framework/gpu-simulator/gpgpu-sim/src/accelwattch/XML_Parse.h`:

```c
enum perf_count_t {
  TOT_INST = 0,              // avg_TOT_INST (Feature 5)
  FP_INT,                    // avg_FP_INT (Feature 3)
  IC_H,                      // Instruction Cache Hits
  IC_M,                      // Instruction Cache Misses
  DC_RH,                     // Data Cache Read Hits
  DC_RM,                     // avg_DC_RM (Feature 9)
  DC_WH,                     // Data Cache Write Hits
  DC_WM,                     // Data Cache Write Misses
  TC_H,                      // Texture Cache Hits
  TC_M,                      // Texture Cache Misses
  CC_H,                      // Constant Cache Hits
  CC_M,                      // Constant Cache Misses
  SHRD_ACC,                  // Shared Memory Accesses
  REG_RD,                    // avg_REG_RD (Feature 14)
  REG_WR,                    // avg_REG_WR (Feature 17)
  NON_REG_OPs,               // Non-Register Operations
  INT_ACC,                   // avg_INT_ACC (Feature 10)
  FP_ACC,                    // FPU Accesses
  DP_ACC,                    // Double Precision Accesses
  INT_MUL24_ACC,             // Integer 24-bit Multiply Accesses
  INT_MUL32_ACC,             // Integer 32-bit Multiply Accesses
  INT_MUL_ACC,               // avg_INT_MUL_ACC (Feature 16)
  INT_DIV_ACC,               // Integer Division Accesses
  FP_MUL_ACC,                // FP Multiply Accesses
  FP_DIV_ACC,                // FP Division Accesses
  FP_SQRT_ACC,               // FP Square Root Accesses
  FP_LG_ACC,                 // FP Logarithm Accesses
  FP_SIN_ACC,                // FP Sine Accesses
  FP_EXP_ACC,                // FP Exponential Accesses
  DP_MUL_ACC,                // DP Multiply Accesses
  DP_DIV_ACC,                // DP Division Accesses
  TENSOR_ACC,                // Tensor Unit Accesses
  TEX_ACC,                   // Texture Unit Accesses
  MEM_RD,                    // Memory Reads
  MEM_WR,                    // Memory Writes
  MEM_PRE,                   // Memory Precharge
  L2_RH,                     // L2 Cache Read Hits
  L2_RM,                     // L2 Cache Read Misses
  L2_WH,                     // L2 Cache Write Hits
  L2_WM,                     // L2 Cache Write Misses
  NOC_A,                     // Network-on-Chip Accesses
  PIPE_A,                    // avg_PIPE_A (Feature 7)
  IDLE_CORE_N,               // avg_IDLE_CORE_N (Feature 18)
  constant_power,            // Constant Power
  NUM_PERFORMANCE_COUNTERS   // Total = 42
};
```

**Used in Classifier:**
- Index 0: TOT_INST → Feature 5 (avg_TOT_INST)
- Index 1: FP_INT → Feature 3 (avg_FP_INT)
- Index 5: DC_RM → Feature 9 (avg_DC_RM)
- Index 13: REG_RD → Feature 14 (avg_REG_RD)
- Index 14: REG_WR → Feature 17 (avg_REG_WR)
- Index 16: INT_ACC → Feature 10 (avg_INT_ACC)
- Index 21: INT_MUL_ACC → Feature 16 (avg_INT_MUL_ACC)
- Index 39: PIPE_A → Feature 7 (avg_PIPE_A)
- Index 40: IDLE_CORE_N → Feature 18 (avg_IDLE_CORE_N)

---

## Power Component Indices (pwr_cmp_t)

From `/accel-sim-framework/gpu-simulator/gpgpu-sim/src/accelwattch/gpgpu_sim_wrapper.cc`:

```c
enum pwr_cmp_t {
  IBP = 0,                   // avg_IBP (Feature 8)
  ICP,                       // Instruction Cache Power
  DCP,                       // Data Cache Power
  TCP,                       // Texture Cache Power
  CCP,                       // Constant Cache Power
  SHRDP,                     // Shared Memory Power
  RFP,                       // avg_RFP (Feature 12)
  INTP,                      // avg_INTP (Feature 11)
  FPUP,                      // FPU Power
  DPUP,                      // DPU Power
  INT_MUL24P,                // Integer 24-bit Multiplier Power
  INT_MUL32P,                // Integer 32-bit Multiplier Power
  INT_MULP,                  // avg_INT_MULP (Feature 15)
  INT_DIVP,                  // Integer Division Power
  FP_MULP,                   // FP Multiplier Power
  FP_DIVP,                   // FP Division Power
  FP_SQRTP,                  // FP Square Root Power
  FP_LGP,                    // FP Logarithm Power
  FP_SINP,                   // FP Sine Power
  FP_EXP,                    // FP Exponential Power
  DP_MULP,                   // DP Multiplier Power
  DP_DIVP,                   // DP Division Power
  TENSORP,                   // Tensor Unit Power
  TEXP,                      // Texture Power
  SCHEDP,                    // avg_SCHEDP (Feature 4)
  L2CP,                      // L2 Cache Power
  MCP,                       // Memory Controller Power
  NOCP,                      // Network-on-Chip Power
  DRAMP,                     // DRAM Power
  PIPEP,                     // avg_PIPEP (Feature 6)
  IDLE_COREP,                // avg_IDLE_COREP (Feature 19)
  CONSTP,                    // Constant Power
  STATICP,                   // Static Power
  SUBCORE_STATICP,           // Sub-core Static Power
  MEM_STATICP,               // Memory Static Power
  NUM_COMPONENTS_MODELLED    // Total = 35
};
```

**Used in Classifier:**
- Index 0: IBP → Feature 8 (avg_IBP)
- Index 6: RFP → Feature 12 (avg_RFP)
- Index 7: INTP → Feature 11 (avg_INTP)
- Index 12: INT_MULP → Feature 15 (avg_INT_MULP)
- Index 25: SCHEDP → Feature 4 (avg_SCHEDP)
- Index 29: PIPEP → Feature 6 (avg_PIPEP)
- Index 30: IDLE_COREP → Feature 19 (avg_IDLE_COREP)

---

## Feature Extraction in subcore_classifier_integration.h

```cpp
// Feature 0: gpu_sim_insn - from simulator (total_insn parameter)
features[0] = static_cast<double>(total_insn);

// Feature 1: gpu_occupancy - calculated from shader stats
features[1] = (active_slots / total_slots) * 100.0;

// Feature 2: gpu_ipc - from simulator
features[2] = total_insn / total_cycles;

// Feature 3: avg_FP_INT
features[3] = get_perf_counter_avg(wrapper, 1);  // FP_INT

// Feature 4: avg_SCHEDP
features[4] = get_power_component_avg(wrapper, 25);  // SCHEDP

// Feature 5: avg_TOT_INST
features[5] = get_perf_counter_avg(wrapper, 0);  // TOT_INST

// Feature 6: avg_PIPEP
features[6] = get_power_component_avg(wrapper, 29);  // PIPEP

// Feature 7: avg_PIPE_A
features[7] = get_perf_counter_avg(wrapper, 39);  // PIPE_A

// Feature 8: avg_IBP
features[8] = get_power_component_avg(wrapper, 0);  // IBP

// Feature 9: avg_DC_RM
features[9] = get_perf_counter_avg(wrapper, 5);  // DC_RM

// Feature 10: avg_INT_ACC
features[10] = get_perf_counter_avg(wrapper, 16);  // INT_ACC

// Feature 11: avg_INTP
features[11] = get_power_component_avg(wrapper, 7);  // INTP

// Feature 12: avg_RFP
features[12] = get_power_component_avg(wrapper, 6);  // RFP

// Feature 13: gpu_sim_cycle - from simulator
features[13] = static_cast<double>(total_cycles);

// Feature 14: avg_REG_RD
features[14] = get_perf_counter_avg(wrapper, 13);  // REG_RD

// Feature 15: avg_INT_MULP
features[15] = get_power_component_avg(wrapper, 12);  // INT_MULP

// Feature 16: avg_INT_MUL_ACC
features[16] = get_perf_counter_avg(wrapper, 21);  // INT_MUL_ACC

// Feature 17: avg_REG_WR
features[17] = get_perf_counter_avg(wrapper, 14);  // REG_WR

// Feature 18: avg_IDLE_CORE_N
features[18] = get_perf_counter_avg(wrapper, 40);  // IDLE_CORE_N

// Feature 19: avg_IDLE_COREP
features[19] = get_power_component_avg(wrapper, 30);  // IDLE_COREP
```

---

## Accessing Metrics in Code

### From gpgpu_sim_wrapper:

```cpp
// Get performance counter average
double metric = power_wrapper->get_kernel_perf_counter_avg(index);

// Get power component average
double metric = power_wrapper->get_kernel_power_component_avg(index);

// Get sample count
int samples = power_wrapper->get_kernel_sample_count();
```

### Safety:
- All getter methods include bounds checking
- Returns 0.0 for invalid indices
- Handles NULL wrapper gracefully
- No file I/O required

---

## Notes

- Indices are stable and defined in header files
- Updates to enums will require corresponding index updates in feature extraction
- Performance counters and power components are accumulated per kernel instance
- Averaging is done automatically by the power wrapper (via `.avg` member)
- Sample count indicates how many cycles were sampled for averaging

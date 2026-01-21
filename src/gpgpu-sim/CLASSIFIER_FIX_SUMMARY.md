# ML Classifier Direct Memory Access - Fix Summary

## Problem
The initial implementation tried to parse the `accelwattch_power_report.log` file to extract power metrics. However:
1. The file may not exist when the decision is made
2. File I/O timing issues can cause parsing failures
3. The metrics are already calculated in memory during execution

## Solution
Modified the approach to access power metrics directly from the `gpgpu_sim_wrapper` object's in-memory data structures instead of parsing a file.

## Changes Made

### 1. **gpgpu_sim_wrapper.h** - Added Public Getter Methods
```cpp
// Getter methods for accessing average kernel performance counters
double get_kernel_perf_counter_avg(int counter_idx) const {
    if (counter_idx >= 0 && counter_idx < (int)kernel_cmp_perf_counters.size()) {
        return kernel_cmp_perf_counters[counter_idx].avg;
    }
    return 0.0;
}

// Getter methods for accessing average kernel power components
double get_kernel_power_component_avg(int comp_idx) const {
    if (comp_idx >= 0 && comp_idx < (int)kernel_cmp_pwr.size()) {
        return kernel_cmp_pwr[comp_idx].avg;
    }
    return 0.0;
}

// Get the number of kernel samples
int get_kernel_sample_count() const {
    return kernel_sample_count;
}
```

These methods safely access the internal `kernel_cmp_perf_counters` and `kernel_cmp_pwr` vectors without requiring file I/O.

### 2. **subcore_classifier_integration.h** - Updated Feature Extractor
- Removed `PowerReportParser` class (file parsing)
- Changed function signature to accept `gpgpu_sim_wrapper*` instead of file path
- Updated feature extraction to use wrapper getter methods directly
- Metrics are now accessed from memory with proper index mapping

### 3. **gpu-sim.h** - Updated Function Signature
```cpp
unsigned decide_subcore_count(unsigned long long total_insn,
                              unsigned long long total_cycles,
                              class gpgpu_sim_wrapper* power_wrapper);
```

### 4. **gpu-sim.cc** - Updated Implementation and Call Site
- `decide_subcore_count()` now receives `gpgpu_sim_wrapper*` instead of file path
- `set_kernel_done()` passes `m_gpgpusim_wrapper` directly to classifier
- No file path construction needed

## Feature Extraction Mapping

The 20 features are extracted as follows:

```
Index  Feature Name           Source                      Method
0      gpu_sim_insn           Simulator state             Direct calculation
1      gpu_occupancy          Shader stats                Direct calculation
2      gpu_ipc                Simulator state             Calculated (insn/cycles)
3      avg_FP_INT             Power metrics               wrapper->get_kernel_perf_counter_avg(1)
4      avg_SCHEDP             Power metrics               wrapper->get_kernel_power_component_avg(25)
5      avg_TOT_INST           Power metrics               wrapper->get_kernel_perf_counter_avg(0)
6      avg_PIPEP              Power metrics               wrapper->get_kernel_power_component_avg(29)
7      avg_PIPE_A             Power metrics               wrapper->get_kernel_perf_counter_avg(39)
8      avg_IBP                Power metrics               wrapper->get_kernel_power_component_avg(0)
9      avg_DC_RM              Power metrics               wrapper->get_kernel_perf_counter_avg(5)
10     avg_INT_ACC            Power metrics               wrapper->get_kernel_perf_counter_avg(16)
11     avg_INTP               Power metrics               wrapper->get_kernel_power_component_avg(7)
12     avg_RFP                Power metrics               wrapper->get_kernel_power_component_avg(6)
13     gpu_sim_cycle          Simulator state             Direct assignment
14     avg_REG_RD             Power metrics               wrapper->get_kernel_perf_counter_avg(13)
15     avg_INT_MULP           Power metrics               wrapper->get_kernel_power_component_avg(12)
16     avg_INT_MUL_ACC        Power metrics               wrapper->get_kernel_perf_counter_avg(21)
17     avg_REG_WR             Power metrics               wrapper->get_kernel_perf_counter_avg(14)
18     avg_IDLE_CORE_N        Power metrics               wrapper->get_kernel_perf_counter_avg(40)
19     avg_IDLE_COREP         Power metrics               wrapper->get_kernel_power_component_avg(30)
```

## Index References

### Performance Counter Indices (perf_count_t enum from XML_Parse.h):
- 0: TOT_INST
- 1: FP_INT
- 5: DC_RM
- 13: REG_RD
- 14: REG_WR
- 16: INT_ACC
- 21: INT_MUL_ACC
- 39: PIPE_A
- 40: IDLE_CORE_N

### Power Component Indices (pwr_cmp_t enum from gpgpu_sim_wrapper.cc):
- 0: IBP (Instruction Buffer Power)
- 6: RFP (Register File Power)
- 7: INTP (Integer Unit Power)
- 12: INT_MULP (Integer Multiplier Power)
- 25: SCHEDP (Scheduler Power)
- 29: PIPEP (Pipeline Power)
- 30: IDLE_COREP (Idle Core Power)

## Benefits

1. **Reliability**: No file I/O required - data is accessed directly from memory
2. **Timing**: Eliminates timing issues - metrics are available immediately
3. **Robustness**: Eliminates race conditions where file hasn't been written yet
4. **Performance**: No file parsing overhead
5. **Direct Access**: Uses established getter methods with bounds checking

## Error Handling

- Null pointer checks on wrapper object
- Bounds checking in getter methods (returns 0.0 for invalid indices)
- Fallback to default subcore count (2) if feature extraction fails
- All metrics default to 0.0 if wrapper is unavailable

## Execution Flow

1. First kernel completes
2. `set_kernel_done()` is called
3. Total instructions and cycles are calculated from simulator state
4. `decide_subcore_count()` is invoked with `m_gpgpusim_wrapper` pointer
5. `SubcoreFeatureExtractor` extracts 20 features:
   - 4 from simulator state (in-memory)
   - 16 from power wrapper (via getter methods)
6. ML classifier makes prediction
7. Probabilities and predicted subcore count are printed
8. Remaining kernels use predicted subcore count

This implementation is more robust and eliminates the file parsing issues while maintaining all the benefits of the ML-based decision system.

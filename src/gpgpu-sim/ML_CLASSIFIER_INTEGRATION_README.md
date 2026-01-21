# ML-Based Sub-Core Classifier Integration

## Overview

This integration replaces the simple IPC-based sub-core count decision with a trained Logistic Regression classifier that uses 20 features extracted from GPU simulation statistics and power reports.

## Files Added/Modified

### New Files:
1. **subcore_classifier.h** - Auto-generated C++ header containing the trained model
   - Location: `/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/`
   - Contains model coefficients, means, standard deviations, and prediction functions
   - Implements Logistic Regression with softmax for 4-class classification (1, 2, 3, or 4 subcores)

2. **subcore_classifier_integration.h** - Integration layer
   - Location: `/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/`
   - Provides `PowerReportParser` class to parse accelwattch_power_report.log
   - Provides `SubcoreFeatureExtractor` class to extract all 20 required features
   - Handles feature extraction from both simulator state and power report file

### Modified Files:
1. **gpu-sim.cc**
   - Modified `decide_subcore_count()` function signature and implementation
   - Updated `set_kernel_done()` to pass power report path to classifier
   - Added includes for classifier headers

2. **gpu-sim.h**
   - Updated `decide_subcore_count()` function declaration

## Feature Extraction

The classifier requires 20 features, which are extracted from two sources:

### From Simulator State (calculated during execution):
1. **gpu_sim_insn** - Total instructions executed in first kernel
2. **gpu_occupancy** - GPU occupancy percentage (active warps / total warp slots)
3. **gpu_ipc** - Instructions per cycle for first kernel
4. **gpu_sim_cycle** - Total simulation cycles for first kernel

### From Power Report (accelwattch_power_report.log):
The following metrics are parsed from the "Kernel Average Power Data:" section:

5. **avg_FP_INT** - Average FP/INT instruction count
6. **avg_SCHEDP** - Average scheduler power
7. **avg_TOT_INST** - Average total instructions
8. **avg_PIPEP** - Average pipeline power
9. **avg_PIPE_A** - Average pipeline activity
10. **avg_IBP** - Average instruction buffer power
11. **avg_DC_RM** - Average data cache read misses
12. **avg_INT_ACC** - Average integer ALU accesses
13. **avg_INTP** - Average integer unit power
14. **avg_RFP** - Average register file power
15. **avg_REG_RD** - Average register reads
16. **avg_INT_MULP** - Average integer multiplier power
17. **avg_INT_MUL_ACC** - Average integer multiplier accesses
18. **avg_REG_WR** - Average register writes
19. **avg_IDLE_CORE_N** - Average number of idle cores
20. **avg_IDLE_COREP** - Average idle core percentage

## Execution Flow

1. **Kernel Execution**: First kernel instance runs normally
2. **Kernel Completion**: When first kernel completes:
   - `set_kernel_done()` is called
   - Detects first kernel completion
   - Collects statistics (instructions, cycles, etc.)
   - Reads power report file: `accelwattch_power_report.log`
3. **Feature Extraction**: 
   - `SubcoreFeatureExtractor` extracts all 20 features
   - Features are standardized using model's scaler parameters
4. **Prediction**:
   - Logistic Regression model computes class probabilities
   - Returns optimal subcore count (1, 2, 3, or 4)
   - Prints detailed prediction output with probabilities
5. **Application**:
   - Predicted subcore count is applied to remaining kernels
   - Schedulers are adjusted accordingly

## Model Information

- **Model Type**: Logistic Regression
- **Number of Features**: 20
- **Number of Classes**: 4 (predicting 1, 2, 3, or 4 subcores)
- **Training**: Model was trained on historical simulation data
- **Feature Scaling**: StandardScaler (Z-score normalization)
- **Output**: Class probabilities via softmax function

## Usage

The integration is automatic. When you run a simulation:

1. The first kernel instance runs completely
2. After completion, the ML classifier analyzes performance
3. The predicted optimal subcore count is applied
4. Detailed output shows:
   - Feature values used for prediction
   - Probability for each subcore count
   - Final predicted subcore count

### Example Output:

```
========================================
FIRST KERNEL INSTANCE COMPLETED!
Kernel UID: 1
Kernel Name: _Z6KernelP4NodePiPbS2_S2_S1_i
Stream ID: 1
End Cycle: 12345
Total Cycles: 12345
========================================

========================================
ML CLASSIFIER PREDICTION WITH PROBABILITIES
========================================
Probabilities for each subcore count:
  1 subcores: 0.1234 (12.3%)
  2 subcores: 0.6543 (65.4%)
  3 subcores: 0.1987 (19.9%)
  4 subcores: 0.0236 (2.4%)
Predicted subcores: 2
========================================

ML Predicted Sub-core Count: 2
```

## Power Report Requirements

The classifier requires that AccelWattch power modeling is enabled and that the power report file exists at the expected location. The file must contain the "Kernel Average Power Data:" section with all required metrics.

If the power report cannot be read or parsed:
- A warning is printed
- The classifier falls back to a default value (2 subcores)
- Simulation continues normally

## Integration Points

### Key Functions:

1. **`decide_subcore_count(total_insn, total_cycles, power_report_path)`**
   - Main entry point for prediction
   - Creates feature extractor
   - Calls classifier with extracted features
   - Returns predicted subcore count

2. **`SubcoreFeatureExtractor::extract_features()`**
   - Extracts all 20 features
   - Combines simulator state and power report data
   - Returns feature array ready for classifier

3. **`PowerReportParser::parse_file()`**
   - Parses accelwattch_power_report.log
   - Extracts avg metrics from kernel section
   - Stores in map for easy lookup

4. **`SubcoreClassifier::predict()` / `predict_proba()`**
   - Core prediction functions from trained model
   - Apply feature scaling
   - Compute logits and softmax probabilities
   - Return predicted class

## Debugging

To debug the classifier:

1. Check feature values printed in prediction output
2. Verify power report file exists and contains expected metrics
3. Check that feature values are reasonable (not NaN, not extreme)
4. Examine class probabilities to see confidence of prediction

## Future Enhancements

Possible improvements:
1. Add support for multiple kernel instance predictions
2. Implement online learning to adapt to workload
3. Add confidence thresholds for fallback behavior
4. Support alternative power report locations/formats
5. Add feature importance analysis
6. Implement ensemble models for better accuracy

## Notes

- The classifier is invoked only after the first kernel instance completes
- All subsequent kernel instances use the predicted subcore count
- The model was trained on historical simulation data with known optimal configurations
- Feature engineering and selection were performed to identify the most predictive 20 features
- The model achieves good accuracy on test data representing diverse workloads

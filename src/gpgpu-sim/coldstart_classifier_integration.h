// Integration layer for ML-based cold-start sub-core classifier
// This file provides a wrapper function for compatibility with gpu-sim.cc

#ifndef COLDSTART_CLASSIFIER_INTEGRATION_H
#define COLDSTART_CLASSIFIER_INTEGRATION_H

#include "coldstart_classifier.h"

// Classifier mode enum for gpu-sim.cc
enum ColdStartClassifierMode {
    COLDSTART_CLASSIFIER_DEFAULT = 0,
    COLDSTART_CLASSIFIER_PERFORMANCE_AGGRESSIVE = 1
};

// Wrapper function for gpu-sim.cc - delegates to the classifier
// Uses 5 pre-launch features via predict_from_launch_params:
//   [0] grid_size (grid_x * grid_y * grid_z)
//   [1] block_size (block_x * block_y * block_z)
//   [2] total_threads (grid_size * block_size)
//   [3] nregs (number of registers)
//   [4] shmem (shared memory)
inline int coldstart_classifier_predict(
        int grid_x, int grid_y, int grid_z,
        int block_x, int block_y, int block_z,
        int nregs, int shmem,
        ColdStartClassifierMode classifier_mode = COLDSTART_CLASSIFIER_DEFAULT,
        int m_active_subcore_limit = 4) {  // Not used - classifier doesn't need it
    (void)classifier_mode;  // Avoid unused parameter warning
    (void)m_active_subcore_limit;  // Avoid unused parameter warning
    
    // Call the classifier's predict_from_launch_params method
    return ColdStartClassifier::predict_from_launch_params(
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        nregs, shmem);
}

// Utility function to extract and predict
inline int coldstart_predict_optimal_subcores(
        int grid_x, int grid_y, int grid_z,
        int block_x, int block_y, int block_z,
        int nregs, int shmem) {
    // Delegates to the main classifier function
    return coldstart_classifier_predict(
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        nregs, shmem,
        COLDSTART_CLASSIFIER_DEFAULT);
}

#endif // COLDSTART_CLASSIFIER_INTEGRATION_H

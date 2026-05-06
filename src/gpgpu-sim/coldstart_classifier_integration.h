// Integration layer for ML-based cold-start sub-core classifier
// This file provides functionality to predict optimal sub-core count from GPU launch parameters
// Uses pre-launch features ONLY (grid/block dims, nregs, shmem)

#ifndef COLDSTART_CLASSIFIER_INTEGRATION_H
#define COLDSTART_CLASSIFIER_INTEGRATION_H

#include <string>
#include <vector>
#include "coldstart_classifier_rf.h"

// Feature extraction class for cold-start predictions
class ColdStartFeatureExtractor {
public:
    // Extract features from launch parameters with occupancy scaling based on current sub-cores
    static bool extract_features(double* features,
                                 int grid_x, int grid_y, int grid_z,
                                 int block_x, int block_y, int block_z,
                                 int nregs, int shmem,
                                 int current_subcores = 4) {
        int grid_size = grid_x * grid_y * grid_z;
        int block_size = block_x * block_y * block_z;
        
        // Feature order MUST match ColdStartClassifier::FEATURE_NAMES:
        //   [0] grid_size
        //   [1] total_threads (grid_size * block_size)
        //   [2] nregs
        //   [3] threads_per_sm (grid_size * block_size / 80.0)
        //   [4] blocks_per_sm (grid_size / 80.0)
        //   [5] reg_footprint (nregs * block_size)
        //   [6] occupancy_proxy (normalized, scaled by current sub-cores)
        
        features[0] = static_cast<double>(grid_size);
        features[1] = static_cast<double>(grid_size) * block_size;
        features[2] = static_cast<double>(nregs);
        features[3] = static_cast<double>(grid_size) * block_size / 80.0;
        features[4] = static_cast<double>(grid_size) / 80.0;
        features[5] = static_cast<double>(nregs) * block_size;
        
        // Compute base occupancy proxy
        double base_occupancy = std::min(static_cast<double>(block_size) / 1024.0, 1.0) *
                                std::min(static_cast<double>(grid_size) / 80.0, 1.0);
        
        // Scale occupancy by inverse of sub-core count: occupancy * (4 / current_subcores)
        // This normalizes occupancy as if all 4 sub-cores were active
        // Clamp current_subcores to valid range [1, 4] for safety
        int clamped_subcores = std::max(1, std::min(4, current_subcores));
        double occupancy_scale = 4.0 / static_cast<double>(clamped_subcores);
        features[6] = base_occupancy * occupancy_scale;
        
        return true;
    }

    // Predict optimal sub-core count from launch parameters
    static int predict_optimal_subcores(
            int grid_x, int grid_y, int grid_z,
            int block_x, int block_y, int block_z,
            int nregs, int shmem,
            int current_subcores = 4) {
        double features[ColdStartClassifier::NUM_FEATURES];
        
        if (!extract_features(features, grid_x, grid_y, grid_z,
                            block_x, block_y, block_z, nregs, shmem,
                            current_subcores)) {
            printf("Warning: Cold-start feature extraction failed, using default sub-core count\n");
            return 2; // Default fallback
        }
        
        // Use the classifier
        int predicted_subcores = ColdStartClassifier::predict(features);
        
        // Print debug info
        printf("\n========================================\n");
        printf("COLD-START ML CLASSIFIER PREDICTION\n");
        printf("========================================\n");
        printf("Launch parameters:\n");
        printf("  Grid: (%d, %d, %d)\n", grid_x, grid_y, grid_z);
        printf("  Block: (%d, %d, %d)\n", block_x, block_y, block_z);
        printf("  Registers: %d, Shared Memory: %d\n", nregs, shmem);
        printf("  Current sub-cores: %d\n", current_subcores);
        printf("\nExtracted features (occupancy scaled by %.2f):\n", 
               4.0 / static_cast<double>(std::max(1, std::min(4, current_subcores))));
        for (int i = 0; i < ColdStartClassifier::NUM_FEATURES; i++) {
            printf("  [%d] %s: %.6f\n", i,
                   ColdStartClassifier::FEATURE_NAMES[i].c_str(),
                   features[i]);
        }
        printf("\nPredicted sub-cores: %d\n", predicted_subcores);
        printf("========================================\n\n");
        
        return predicted_subcores;
    }
};

// Wrapper function for backward compatibility with existing gpu-sim.cc code
enum ColdStartClassifierMode {
    COLDSTART_CLASSIFIER_DEFAULT = 0,
    COLDSTART_CLASSIFIER_PERFORMANCE_AGGRESSIVE = 1
};

inline int coldstart_classifier_predict(
        int grid_x, int grid_y, int grid_z,
        int block_x, int block_y, int block_z,
        int nregs, int shmem,
        ColdStartClassifierMode mode = COLDSTART_CLASSIFIER_DEFAULT,
        int current_subcores = 4) {
    (void)mode; // Mode handling can be added later if needed
    return ColdStartFeatureExtractor::predict_optimal_subcores(
        grid_x, grid_y, grid_z, block_x, block_y, block_z, nregs, shmem,
        current_subcores);
}

#endif // COLDSTART_CLASSIFIER_INTEGRATION_H

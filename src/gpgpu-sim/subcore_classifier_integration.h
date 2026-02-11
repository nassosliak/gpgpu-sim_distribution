// Integration layer for ML-based sub-core classifier
// This file provides functionality to extract features from GPU simulator
// and power statistics for the trained classifier

#ifndef SUBCORE_CLASSIFIER_INTEGRATION_H
#define SUBCORE_CLASSIFIER_INTEGRATION_H

#include <string>
#include "subcore_classifier.h"
#include "gpu-sim.h"
#include "../accelwattch/gpgpu_sim_wrapper.h"

// Feature extraction class
class SubcoreFeatureExtractor {
private:
    const gpgpu_sim* m_gpu;
    const shader_core_config* m_shader_config;
    const shader_core_stats* m_shader_stats;
    
public:
    SubcoreFeatureExtractor(const gpgpu_sim* gpu,
                           const shader_core_config* shader_config,
                           const shader_core_stats* shader_stats)
        : m_gpu(gpu), m_shader_config(shader_config), m_shader_stats(shader_stats) {}
    
    // Extract all 4 features required by the classifier
    bool extract_features(double* features, 
                         class gpgpu_sim_wrapper* power_wrapper) {
        
        if (!power_wrapper) {
            printf("Warning: Power wrapper is NULL, cannot extract power metrics\n");
            return false;
        }
        
          int sample_count = power_wrapper->get_kernel_sample_count();
          if (sample_count <= 0) {
              printf("Warning: No power samples collected yet\n");
              return false;
          }

          // Feature order MUST match DecisionTree::FEATURE_NAMES:
          //   [0] gpu_ipc, [1] avg_RFP, [2] avg_INTP, [3] avg_INT_ACC
          //
          // Power component indices (from pwr_cmp_t enum):
          //   RFP=6, INTP=7
          // Performance counter indices (from perf_count_t enum):
          //   INT_ACC=16

          // Feature 0: gpu_ipc
          features[0] = (m_gpu->gpu_sim_cycle > 0)
                      ? static_cast<double>(m_gpu->gpu_sim_insn) /
                          static_cast<double>(m_gpu->gpu_sim_cycle)
                      : 0.0;

          // Feature 1: avg_RFP (power component index RFP = 6)
          features[1] = power_wrapper->get_kernel_power_component_avg(6);

          // Feature 2: avg_INTP (power component index INTP = 7)
          features[2] = power_wrapper->get_kernel_power_component_avg(7);

          // Feature 3: avg_INT_ACC (performance counter index INT_ACC = 16)
          features[3] = power_wrapper->get_kernel_perf_counter_avg(16);
        
        return true;
    }
    
public:
    // Convenience method to predict optimal subcore count
    int predict_optimal_subcores(class gpgpu_sim_wrapper* power_wrapper) {
        double features[DecisionTree::NUM_FEATURES];
        
        if (!extract_features(features, power_wrapper)) {
            printf("Warning: Feature extraction failed, using default subcore count\n");
            return 2; // Default fallback
        }
        
        // Use the classifier
        int predicted_subcores = DecisionTree::predict(features);
        
        // Print debug info
        printf("\n========================================\n");
        printf("ML CLASSIFIER PREDICTION\n");
        printf("========================================\n");
        printf("Feature values:\n");
        for (int i = 0; i < DecisionTree::NUM_FEATURES; i++) {
            printf("  %s: %.6f\n", 
                   DecisionTree::FEATURE_NAMES[i].c_str(), 
                   features[i]);
        }
        printf("Predicted subcores: %d\n", predicted_subcores);
        printf("========================================\n\n");
        
        return predicted_subcores;
    }
    
    // Version with probability output
    int predict_optimal_subcores_with_proba(class gpgpu_sim_wrapper* power_wrapper) {
        double features[DecisionTree::NUM_FEATURES];
        double probabilities[DecisionTree::NUM_CLASSES];
        
        if (!extract_features(features, power_wrapper)) {
            printf("Warning: Feature extraction failed, using default subcore count\n");
            return 2; // Default fallback
        }
        
        // Use the classifier with probabilities
        int predicted_subcores = DecisionTree::predict_proba(features, probabilities);
        
        // Print debug info with probabilities
        printf("\n========================================\n");
        printf("ML CLASSIFIER PREDICTION WITH PROBABILITIES\n");
        printf("========================================\n");
        printf("Feature values:\n");
        for (int i = 0; i < DecisionTree::NUM_FEATURES; i++) {
            printf("  [%2d] %s: %.6f\n", i,
                   DecisionTree::FEATURE_NAMES[i].c_str(), 
                   features[i]);
        }
        printf("\nProbabilities for each subcore count:\n");
        for (int c = 0; c < DecisionTree::NUM_CLASSES; c++) {
            printf("  %d subcores: %.4f (%.1f%%)\n", 
                   DecisionTree::CLASSES[c],
                   probabilities[c],
                   probabilities[c] * 100.0);
        }
        printf("Predicted subcores: %d\n", predicted_subcores);
        printf("========================================\n\n");
        
        return predicted_subcores;
    }
};

#endif // SUBCORE_CLASSIFIER_INTEGRATION_H

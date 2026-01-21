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
    
    // Extract all 20 features required by the classifier
    bool extract_features(double* features, 
                         class gpgpu_sim_wrapper* power_wrapper) {
        
        if (!power_wrapper) {
            printf("Warning: Power wrapper is NULL, cannot extract power metrics\n");
            return false;
        }
        
        // Debug: Check sample count
        int sample_count = power_wrapper->get_kernel_sample_count();
        printf("DEBUG: Power wrapper sample count = %d\n", sample_count);
        
        // Feature 0: gpu_sim_insn - use gpgpu_sim member variable
        features[0] = static_cast<double>(m_gpu->gpu_sim_insn);
        
        // Feature 1: gpu_occupancy - use gpu_occupancy from gpgpu_sim
        features[1] = m_gpu->gpu_occupancy.get_occ_fraction() * 100.0;
        
        // Feature 2: gpu_ipc - calculate from gpgpu_sim member variables
        features[2] = (m_gpu->gpu_sim_cycle > 0) ? 
                      static_cast<double>(m_gpu->gpu_sim_insn) / static_cast<double>(m_gpu->gpu_sim_cycle) : 0.0;
        
        // Access power metrics directly from power wrapper
        // The power wrapper has kernel_cmp_perf_counters which contains all the metrics
        // indexed by the perf_count_t enum from XML_Parse.h
        
        // Feature 3: avg_FP_INT - performance counter index FP_INT (1)
        features[3] = get_perf_counter_avg(power_wrapper, 1);  // FP_INT
        printf("DEBUG: Feature 3 (avg_FP_INT) from perf[1] = %.6f\n", features[3]);
        
        // Feature 4: avg_SCHEDP - scheduler power (power component index SCHEDP = 25)
        features[4] = get_power_component_avg(power_wrapper, 25);  // SCHEDP
        
        // Feature 5: avg_TOT_INST - performance counter index TOT_INST (0)
        features[5] = get_perf_counter_avg(power_wrapper, 0);  // TOT_INST
        
        // Feature 6: avg_PIPEP - pipeline power (power component index PIPEP = 29)
        features[6] = get_power_component_avg(power_wrapper, 29);  // PIPEP
        
        // Feature 7: avg_PIPE_A - performance counter index PIPE_A (39)
        features[7] = get_perf_counter_avg(power_wrapper, 39);  // PIPE_A
        
        // Feature 8: avg_IBP - instruction buffer power (power component index IBP = 0)
        features[8] = get_power_component_avg(power_wrapper, 0);  // IBP
        
        // Feature 9: avg_DC_RM - performance counter index DC_RM (5)
        features[9] = get_perf_counter_avg(power_wrapper, 5);  // DC_RM
        
        // Feature 10: avg_INT_ACC - performance counter index INT_ACC (16)
        features[10] = get_perf_counter_avg(power_wrapper, 16);  // INT_ACC
        
        // Feature 11: avg_INTP - integer power (power component index INTP = 7)
        features[11] = get_power_component_avg(power_wrapper, 7);  // INTP
        
        // Feature 12: avg_RFP - register file power (power component index RFP = 6)
        features[12] = get_power_component_avg(power_wrapper, 6);  // RFP
        
        // Feature 13: gpu_sim_cycle - use gpgpu_sim member variable
        features[13] = static_cast<double>(m_gpu->gpu_sim_cycle);
        
        // Feature 14: avg_REG_RD - performance counter index REG_RD (13)
        features[14] = get_perf_counter_avg(power_wrapper, 13);  // REG_RD
        
        // Feature 15: avg_INT_MULP - integer multiplier power (power component index INT_MULP = 12)
        features[15] = get_power_component_avg(power_wrapper, 12);  // INT_MULP
        
        // Feature 16: avg_INT_MUL_ACC - performance counter index INT_MUL_ACC (21)
        features[16] = get_perf_counter_avg(power_wrapper, 21);  // INT_MUL_ACC
        
        // Feature 17: avg_REG_WR - performance counter index REG_WR (14)
        features[17] = get_perf_counter_avg(power_wrapper, 14);  // REG_WR
        
        // Feature 18: avg_IDLE_CORE_N - performance counter index IDLE_CORE_N (40)
        features[18] = get_perf_counter_avg(power_wrapper, 40);  // IDLE_CORE_N
        
        // Feature 19: avg_IDLE_COREP - idle core power (power component index IDLE_COREP = 30)
        features[19] = get_power_component_avg(power_wrapper, 30);  // IDLE_COREP
        
        return true;
    }
    
private:
    // Helper function to get average performance counter from power wrapper
    double get_perf_counter_avg(class gpgpu_sim_wrapper* wrapper, int perf_counter_idx) {
        if (!wrapper) {
            printf("WARNING: wrapper is NULL for perf counter %d\n", perf_counter_idx);
            return 0.0;
        }
        double val = wrapper->get_kernel_perf_counter_avg(perf_counter_idx);
        // Debug first few calls
        if (perf_counter_idx <= 5) {
            printf("DEBUG: perf_counter[%d] = %.6f\n", perf_counter_idx, val);
        }
        return val;
    }
    
    // Helper function to get average power component from power wrapper
    double get_power_component_avg(class gpgpu_sim_wrapper* wrapper, int power_comp_idx) {
        if (!wrapper) {
            printf("WARNING: wrapper is NULL for power component %d\n", power_comp_idx);
            return 0.0;
        }
        double val = wrapper->get_kernel_power_component_avg(power_comp_idx);
        // Debug first few calls
        if (power_comp_idx <= 7) {
            printf("DEBUG: power_component[%d] = %.6f\n", power_comp_idx, val);
        }
        return val;
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

// Integration layer for ML-based sub-core classifier
// This file provides functionality to extract features from GPU simulator
// and power statistics for the trained classifier

#ifndef SUBCORE_CLASSIFIER_INTEGRATION_H
#define SUBCORE_CLASSIFIER_INTEGRATION_H

#include <string>
#include <vector>
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

    // Extract all features required by the classifier
    bool extract_features(double* features,
                         class gpgpu_sim_wrapper* power_wrapper) {

        (void)m_shader_config;
        (void)m_shader_stats;

        if (!power_wrapper) {
            printf("Warning: Power wrapper is NULL, cannot extract power metrics\n");
            return false;
        }

        int sample_count = power_wrapper->get_kernel_sample_count();
        if (sample_count <= 0) {
            printf("Warning: No power samples collected yet\n");
            return false;
        }

        // Feature order MUST match SubcoreRandomForestClassifier::FEATURE_NAMES:
        //   [0] cmp_gpu_ipc, [1] cmp_gpu_occupancy, [2] cmp_gpu_sim_insn,
        //   [3] cmp_avg_IBP, [4] cmp_avg_RFP, [5] cmp_avg_INTP, [6] cmp_avg_INT_MULP,
        //   [7] cmp_avg_SCHEDP, [8] cmp_avg_TOT_INST, [9] cmp_avg_FP_INT,
        //   [10] cmp_avg_REG_RD, [11] cmp_avg_REG_WR, [12] cmp_avg_INT_ACC,
        //   [13] cmp_avg_INT_MUL_ACC, [14] cmp_avg_threads_per_warp
        //
        // Power component indices:
        //   IBP=0, RFP=6, INTP=7, INT_MULP=12, SCHEDP=24
        // Performance counter indices:
        //   TOT_INST=0, FP_INT=1, REG_RD=13, REG_WR=14, INT_ACC=16, INT_MUL_ACC=21

        // Feature 0: cmp_gpu_ipc
        features[0] = (m_gpu->gpu_sim_cycle > 0)
                      ? static_cast<double>(m_gpu->gpu_sim_insn) /
                            static_cast<double>(m_gpu->gpu_sim_cycle)
                      : 0.0;

        // Feature 1: cmp_gpu_occupancy
        features[1] = (m_gpu->gpu_occupancy.aggregate_theoretical_warp_slots > 0)
                      ? m_gpu->gpu_occupancy.get_occ_fraction() * 100.0
                      : 0.0;

        // Feature 2: cmp_gpu_sim_insn
        features[2] = static_cast<double>(m_gpu->gpu_sim_insn);

        // Feature 3: cmp_avg_IBP (power component index IBP = 0)
        features[3] = power_wrapper->get_kernel_power_component_avg(0);

        // Feature 4: cmp_avg_RFP (power component index RFP = 6)
        features[4] = power_wrapper->get_kernel_power_component_avg(6);

        // Feature 5: cmp_avg_INTP (power component index INTP = 7)
        features[5] = power_wrapper->get_kernel_power_component_avg(7);

        // Feature 6: cmp_avg_INT_MULP (power component index INT_MULP = 12)
        features[6] = power_wrapper->get_kernel_power_component_avg(12);

        // Feature 7: cmp_avg_SCHEDP (power component index SCHEDP = 24)
        features[7] = power_wrapper->get_kernel_power_component_avg(24);

        // Feature 8: cmp_avg_TOT_INST (performance counter index TOT_INST = 0)
        features[8] = power_wrapper->get_kernel_perf_counter_avg(0);

        // Feature 9: cmp_avg_FP_INT (performance counter index FP_INT = 1)
        features[9] = power_wrapper->get_kernel_perf_counter_avg(1);

        // Feature 10: cmp_avg_REG_RD (performance counter index REG_RD = 13)
        features[10] = power_wrapper->get_kernel_perf_counter_avg(13);

        // Feature 11: cmp_avg_REG_WR (performance counter index REG_WR = 14)
        features[11] = power_wrapper->get_kernel_perf_counter_avg(14);

        // Feature 12: cmp_avg_INT_ACC (performance counter index INT_ACC = 16)
        features[12] = power_wrapper->get_kernel_perf_counter_avg(16);

        // Feature 13: cmp_avg_INT_MUL_ACC (performance counter index INT_MUL_ACC = 21)
        features[13] = power_wrapper->get_kernel_perf_counter_avg(21);

        // Feature 14: cmp_avg_threads_per_warp
        features[14] = power_wrapper->get_kernel_avg_threads_per_warp();

        return true;
    }

public:
    // Convenience method to predict optimal subcore count
    int predict_optimal_subcores(class gpgpu_sim_wrapper* power_wrapper) {
        double features[SubcoreRandomForestClassifier::NUM_FEATURES];

        if (!extract_features(features, power_wrapper)) {
            printf("Warning: Feature extraction failed, using default subcore count\n");
            return 2; // Default fallback
        }

        // Use the classifier
        int predicted_subcores = SubcoreRandomForestClassifier::predict(features);

        // Print debug info
        printf("\n========================================\n");
        printf("ML CLASSIFIER PREDICTION\n");
        printf("========================================\n");
        printf("Feature values:\n");
        for (int i = 0; i < SubcoreRandomForestClassifier::NUM_FEATURES; i++) {
            printf("  %s: %.6f\n",
                   SubcoreRandomForestClassifier::FEATURE_NAMES[i].c_str(),
                   features[i]);
        }
        printf("Predicted subcores: %d\n", predicted_subcores);
        printf("========================================\n\n");

        return predicted_subcores;
    }

    // Version with probability output
    int predict_optimal_subcores_with_proba(class gpgpu_sim_wrapper* power_wrapper) {
        double features[SubcoreRandomForestClassifier::NUM_FEATURES];
        double probabilities[SubcoreRandomForestClassifier::NUM_CLASSES];

        if (!extract_features(features, power_wrapper)) {
            printf("Warning: Feature extraction failed, using default subcore count\n");
            return 2; // Default fallback
        }

        // Use the classifier with probabilities
        int predicted_subcores = SubcoreRandomForestClassifier::predict_proba(features, probabilities);

        // Print debug info with probabilities
        printf("\n========================================\n");
        printf("ML CLASSIFIER PREDICTION WITH PROBABILITIES\n");
        printf("========================================\n");
        printf("Feature values:\n");
        for (int i = 0; i < SubcoreRandomForestClassifier::NUM_FEATURES; i++) {
            printf("  [%2d] %s: %.6f\n", i,
                   SubcoreRandomForestClassifier::FEATURE_NAMES[i].c_str(),
                   features[i]);
        }
        printf("\nProbabilities for each subcore count:\n");
        for (int c = 0; c < SubcoreRandomForestClassifier::NUM_CLASSES; c++) {
            printf("  %d subcores: %.4f (%.1f%%)\n",
                   SubcoreRandomForestClassifier::CLASSES[c],
                   probabilities[c],
                   probabilities[c] * 100.0);
        }
        printf("Predicted subcores: %d\n", predicted_subcores);
        printf("========================================\n\n");

        return predicted_subcores;
    }
};

#endif // SUBCORE_CLASSIFIER_INTEGRATION_H

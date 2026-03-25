// Integration layer for performance-aggressive sub-core classifier

#ifndef PERFORMANCE_SUBCORE_CLASSIFIER_INTEGRATION_H
#define PERFORMANCE_SUBCORE_CLASSIFIER_INTEGRATION_H

#include <string>
#include "gpu-sim.h"
#include "performance_subcore_classifier.h"
#include "../accelwattch/gpgpu_sim_wrapper.h"

class PerformanceSubcoreFeatureExtractor {
 private:
  const gpgpu_sim* m_gpu;
  const shader_core_config* m_shader_config;
  const shader_core_stats* m_shader_stats;

 public:
  PerformanceSubcoreFeatureExtractor(const gpgpu_sim* gpu,
                                     const shader_core_config* shader_config,
                                     const shader_core_stats* shader_stats)
      : m_gpu(gpu),
        m_shader_config(shader_config),
        m_shader_stats(shader_stats) {}

  bool extract_features(double* features, class gpgpu_sim_wrapper* power_wrapper,
                        int nregs) {
    (void)m_shader_config;
    (void)m_shader_stats;

    if (!power_wrapper) {
      printf("Warning: Power wrapper is NULL, cannot extract performance-mode metrics\n");
      return false;
    }

    int sample_count = power_wrapper->get_kernel_sample_count();
    if (sample_count <= 0) {
      printf("Warning: No power/perf samples collected yet\n");
      return false;
    }

    // Feature order MUST match performance_classifier::DecisionTree::FEATURE_NAMES:
    // [0] gpu_ipc, [1] gpu_occupancy, [2] gpu_sim_insn, [3] gpu_sim_cycle,
    // [4] nregs, [5] avg_INT_MULP, [6] avg_DC_RM, [7] avg_INT_MUL_ACC
    // Indices:
    //   power component INT_MULP = 12 (see accelwattch/gpgpu_sim_wrapper.cc)
    //   perf counter DC_RM = 5, INT_MUL_ACC = 21 (see accelwattch/XML_Parse.h)
    features[0] = (m_gpu->gpu_sim_cycle > 0)
                      ? static_cast<double>(m_gpu->gpu_sim_insn) /
                            static_cast<double>(m_gpu->gpu_sim_cycle)
                      : 0.0;

    features[1] = (m_gpu->gpu_occupancy.aggregate_theoretical_warp_slots > 0)
                      ? m_gpu->gpu_occupancy.get_occ_fraction() * 100.0
                      : 0.0;

    features[2] = static_cast<double>(m_gpu->gpu_sim_insn);
    features[3] = static_cast<double>(m_gpu->gpu_sim_cycle);
    features[4] = static_cast<double>(nregs);
    features[5] = power_wrapper->get_kernel_power_component_avg(12);
    features[6] = power_wrapper->get_kernel_perf_counter_avg(5);
    features[7] = power_wrapper->get_kernel_perf_counter_avg(21);

    return true;
  }

  int predict_optimal_subcores_with_proba(class gpgpu_sim_wrapper* power_wrapper,
                                          int nregs) {
    double features[performance_classifier::DecisionTree::NUM_FEATURES];
    double probabilities[performance_classifier::DecisionTree::NUM_CLASSES];

    if (!extract_features(features, power_wrapper, nregs)) {
      printf("Warning: Performance-mode feature extraction failed, using default subcore count\n");
      return 2;
    }

    int predicted_subcores =
        performance_classifier::DecisionTree::predict_proba(features,
                                                            probabilities);

    printf("\n========================================\n");
    printf("PERFORMANCE-AGGRESSIVE CLASSIFIER PREDICTION\n");
    printf("========================================\n");
    printf("Feature values:\n");
    for (int i = 0; i < performance_classifier::DecisionTree::NUM_FEATURES;
         i++) {
      printf("  [%2d] %s: %.6f\n", i,
             performance_classifier::DecisionTree::FEATURE_NAMES[i].c_str(),
             features[i]);
    }
    printf("\nProbabilities for each subcore count:\n");
    for (int c = 0; c < performance_classifier::DecisionTree::NUM_CLASSES;
         c++) {
      printf("  %d subcores: %.4f (%.1f%%)\n",
             performance_classifier::DecisionTree::CLASSES[c], probabilities[c],
             probabilities[c] * 100.0);
    }
    printf("Predicted subcores: %d\n", predicted_subcores);
    printf("========================================\n\n");

    return predicted_subcores;
  }
};

#endif  // PERFORMANCE_SUBCORE_CLASSIFIER_INTEGRATION_H

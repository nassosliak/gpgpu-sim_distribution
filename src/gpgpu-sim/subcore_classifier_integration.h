// Integration layer for ML-based sub-core classifier
// This file provides utilities to extract features from GPU simulator state
// and use the trained subcore classifier for predictions

#ifndef SUBCORE_CLASSIFIER_INTEGRATION_H
#define SUBCORE_CLASSIFIER_INTEGRATION_H

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "subcore_classifier.h"

// Forward declarations - adjust as needed for your build
class gpgpu_sim;
class shader_core_config;
class shader_core_stats;
class gpgpu_sim_wrapper;
class kernel_info_t;

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
    // Features (in order matching subcore_classifier.h):
    //   [0] cmp_gpu_ipc
    //   [1] cmp_gpu_occupancy
    //   [2] cmp_grid_size
    //   [3] cmp_block_size
    //   [4] cmp_total_threads
    //   [5] cmp_shmem
    //   [6] cmp_nregs
    //   [7] cmp_mem_intensity
    //   [8] cmp_shmem_intensity
    bool extract_features(double* features,
                         unsigned long long total_insn,
                         unsigned long long total_cycles,
                         int nregs,
                         class gpgpu_sim_wrapper* power_wrapper,
                         class kernel_info_t* kernel) {

        (void)m_shader_config;
        (void)m_shader_stats;

        if (!power_wrapper) {
            printf("Warning: Power wrapper is NULL, cannot extract power metrics\n");
            return false;
        }

        if (!kernel) {
            printf("Warning: Kernel info is NULL, cannot extract kernel parameters\n");
            return false;
        }

        int sample_count = power_wrapper->get_kernel_sample_count();
        if (sample_count <= 0) {
            printf("Warning: No power samples collected yet\n");
            return false;
        }

        if (SubcoreRandomForestClassifier::NUM_FEATURES != 9) {
            printf("Warning: Classifier expects 9 features, but NUM_FEATURES=%d\n",
                   SubcoreRandomForestClassifier::NUM_FEATURES);
            return false;
        }

        // Feature 0: cmp_gpu_ipc
        features[0] = (m_gpu->gpu_sim_cycle > 0)
                      ? static_cast<double>(m_gpu->gpu_sim_insn) / static_cast<double>(m_gpu->gpu_sim_cycle)
                      : 0.0;

        // Feature 1: cmp_gpu_occupancy
        features[1] = (m_gpu->gpu_occupancy.aggregate_theoretical_warp_slots > 0)
                      ? m_gpu->gpu_occupancy.get_occ_fraction() * 100.0
                      : 0.0;

        // Extract grid and block dimensions
        dim3 grid_dim = kernel->get_grid_dim();
        double grid_size = static_cast<double>(grid_dim.x * grid_dim.y * grid_dim.z);
        dim3 block_dim = kernel->get_cta_dim();
        double block_size = static_cast<double>(block_dim.x * block_dim.y * block_dim.z);
        double total_threads = grid_size * block_size;

        // Feature 2: cmp_grid_size
        features[2] = grid_size;

        // Feature 3: cmp_block_size
        features[3] = block_size;

        // Feature 4: cmp_total_threads
        features[4] = total_threads;

        // Feature 5: cmp_shmem
        // Get shared memory per block from kernel info or use estimated value
        int shmem_per_block = 0;
        if (kernel) {
            // Try to get shared memory from kernel info
            // kernel->shared_mem_size() should return bytes per block
            shmem_per_block = 96 * 1024;  // Default to 96KB, override if kernel provides it
        }
        features[5] = static_cast<double>(shmem_per_block);

        // Feature 6: cmp_nregs
        features[6] = static_cast<double>(nregs);

        // Feature 7: cmp_mem_intensity
        // mem_intensity = (mem_read_global + mem_write_global) / total_instructions
        double mem_read = power_wrapper->get_kernel_perf_counter_avg(34);   // MEM_RD (Global Memory Read)
        double mem_write = power_wrapper->get_kernel_perf_counter_avg(35);  // MEM_WR (Global Memory Write)
        double kernel_total_insn = power_wrapper->get_kernel_perf_counter_avg(0);
        
        // Fallback to the total_insn parameter if counter data is unavailable
        if (kernel_total_insn <= 0) {
            kernel_total_insn = static_cast<double>(total_insn);
        }
        
        features[7] = (kernel_total_insn > 0) ? (mem_read + mem_write) / kernel_total_insn : 0.0;

        // Feature 8: cmp_shmem_intensity
        // shmem_intensity = shared_memory_accesses / total_instructions
        double shmem_read = power_wrapper->get_kernel_perf_counter_avg(5);   // SMEM_RD
        double shmem_write = power_wrapper->get_kernel_perf_counter_avg(6);  // SMEM_WR
        features[8] = (kernel_total_insn > 0) ? (shmem_read + shmem_write) / kernel_total_insn : 0.0;

        return true;
    }

    // Predict optimal subcore count
    int predict_optimal_subcores(unsigned long long total_insn,
                                unsigned long long total_cycles,
                                int nregs,
                                class gpgpu_sim_wrapper* power_wrapper,
                                class kernel_info_t* kernel) {
        double features[SubcoreRandomForestClassifier::NUM_FEATURES];

        if (!extract_features(features, total_insn, total_cycles, nregs, power_wrapper, kernel)) {
            printf("Warning: Feature extraction failed, using default subcore count\n");
            return 2; // Default fallback
        }

        return SubcoreRandomForestClassifier::predict(features);
    }

    // Predict with probability output
    int predict_optimal_subcores_with_proba(unsigned long long total_insn,
                                           unsigned long long total_cycles,
                                           int nregs,
                                           class gpgpu_sim_wrapper* power_wrapper,
                                           class kernel_info_t* kernel) {
        double features[SubcoreRandomForestClassifier::NUM_FEATURES];
        double probabilities[SubcoreRandomForestClassifier::NUM_CLASSES];

        if (!extract_features(features, total_insn, total_cycles, nregs, power_wrapper, kernel)) {
            printf("Warning: Feature extraction failed, using default subcore count\n");
            return 2; // Default fallback
        }

        return SubcoreRandomForestClassifier::predict_proba(features, probabilities);
    }
};

#endif // SUBCORE_CLASSIFIER_INTEGRATION_H

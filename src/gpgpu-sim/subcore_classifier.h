// Auto-generated sub-core classifier for C++ integration
// Generated from compact_classifier.py
// Model: Decision Tree with 4 features

#ifndef SUBCORE_CLASSIFIER_H
#define SUBCORE_CLASSIFIER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

class DecisionTree {
public:
    // Model configuration
    static constexpr int NUM_FEATURES = 4;
    static constexpr int NUM_CLASSES = 4;
    static constexpr int NUM_NODES = 57;
    
    // Feature names (for reference)
    static const std::vector<std::string> FEATURE_NAMES;
    
    // Scaler parameters
    static const double SCALER_MEAN[NUM_FEATURES];
    static const double SCALER_STD[NUM_FEATURES];
    
    // Decision tree structure
    static const int CHILDREN_LEFT[NUM_NODES];
    static const int CHILDREN_RIGHT[NUM_NODES];
    static const int FEATURE[NUM_NODES];
    static const double THRESHOLD[NUM_NODES];
    static const double VALUE[NUM_NODES][NUM_CLASSES];
    static const int CLASSES[NUM_CLASSES];
    
    // Predict optimal number of sub-cores
    static int predict(const double* features) {
        // Standardize features
        double scaled_features[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            scaled_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_STD[i];
        }
        
        // Traverse decision tree
        int node = 0;
        while (CHILDREN_LEFT[node] != -1) {  // Not a leaf
            if (scaled_features[FEATURE[node]] <= THRESHOLD[node]) {
                node = CHILDREN_LEFT[node];
            } else {
                node = CHILDREN_RIGHT[node];
            }
        }
        
        // Find class with maximum value at leaf node
        int best_class_idx = 0;
        double max_value = VALUE[node][0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (VALUE[node][c] > max_value) {
                max_value = VALUE[node][c];
                best_class_idx = c;
            }
        }
        
        return CLASSES[best_class_idx];
    }
    
    // Predict with probabilities
    static int predict_proba(const double* features, double* probabilities) {
        // Standardize features
        double scaled_features[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            scaled_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_STD[i];
        }
        
        // Traverse decision tree
        int node = 0;
        while (CHILDREN_LEFT[node] != -1) {  // Not a leaf
            if (scaled_features[FEATURE[node]] <= THRESHOLD[node]) {
                node = CHILDREN_LEFT[node];
            } else {
                node = CHILDREN_RIGHT[node];
            }
        }
        
        // Calculate probabilities from leaf node values
        double total = 0.0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            total += VALUE[node][c];
        }
        
        int best_class_idx = 0;
        double max_prob = 0.0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            probabilities[c] = (total > 0) ? (VALUE[node][c] / total) : 0.0;
            if (probabilities[c] > max_prob) {
                max_prob = probabilities[c];
                best_class_idx = c;
            }
        }
        
        return CLASSES[best_class_idx];
    }
};

// Feature names
const std::vector<std::string> DecisionTree::FEATURE_NAMES = {
    "gpu_ipc",
    "avg_RFP",
    "avg_INTP",
    "avg_INT_ACC"
};

const double DecisionTree::SCALER_MEAN[NUM_FEATURES] = {
    1.0632982739e+03, 7.6380881069e+00, 2.3746736068e+00, 2.2709433382e+06
};

const double DecisionTree::SCALER_STD[NUM_FEATURES] = {
    1.4521198421e+03, 1.0756542552e+01, 4.2171422335e+00, 4.0329273964e+06
};

const int DecisionTree::CHILDREN_LEFT[NUM_NODES] = {
    1, 2, -1, -1, 5, 6, 7, 8, -1, -1,
    11, 12, -1, 14, -1, -1, 17, 18, -1, -1,
    -1, 22, 23, 24, 25, -1, -1, -1, -1, 30,
    31, 32, 33, -1, -1, 36, -1, -1, -1, 40,
    41, 42, -1, -1, -1, -1, 47, 48, 49, -1,
    51, -1, -1, -1, 55, -1, -1
};

const int DecisionTree::CHILDREN_RIGHT[NUM_NODES] = {
    4, 3, -1, -1, 46, 21, 10, 9, -1, -1,
    16, 13, -1, 15, -1, -1, 20, 19, -1, -1,
    -1, 29, 28, 27, 26, -1, -1, -1, -1, 39,
    38, 35, 34, -1, -1, 37, -1, -1, -1, 45,
    44, 43, -1, -1, -1, -1, 54, 53, 50, -1,
    52, -1, -1, -1, 56, -1, -1
};

const int DecisionTree::FEATURE[NUM_NODES] = {
    2, 1, -2, -2, 0, 1, 2, 2, -2, -2,
    1, 2, -2, 1, -2, -2, 0, 1, -2, -2,
    -2, 0, 3, 0, 1, -2, -2, -2, -2, 2,
    1, 1, 3, -2, -2, 0, -2, -2, -2, 0,
    3, 1, -2, -2, -2, -2, 1, 0, 1, -2,
    1, -2, -2, -2, 2, -2, -2
};

const double DecisionTree::THRESHOLD[NUM_NODES] = {
    -5.4394060373e-01, -6.6200664639e-01, -2.0000000000e+00, -2.0000000000e+00,
    5.0958639383e-01, -6.8567955494e-01, -5.3914690018e-01, -5.4191464186e-01,
    -2.0000000000e+00, -2.0000000000e+00, -6.9096028805e-01, -5.3417289257e-01,
    -2.0000000000e+00, -6.9144883752e-01, -2.0000000000e+00, -2.0000000000e+00,
    -6.9920995831e-01, -6.8917357922e-01, -2.0000000000e+00, -2.0000000000e+00,
    -2.0000000000e+00, -6.0809311271e-01, -5.3859901428e-01, -6.9061222672e-01,
    -6.5189611912e-01, -2.0000000000e+00, -2.0000000000e+00, -2.0000000000e+00,
    -2.0000000000e+00, -1.6087045521e-01, 3.9669455588e-01, -5.6997245550e-01,
    -4.3344838917e-01, -2.0000000000e+00, -2.0000000000e+00, -2.2897119820e-01,
    -2.0000000000e+00, -2.0000000000e+00, -2.0000000000e+00, 1.3467444107e-01,
    1.2498778850e-01, -1.0885543376e-01, -2.0000000000e+00, -2.0000000000e+00,
    -2.0000000000e+00, -2.0000000000e+00, 2.7499181032e+00, 1.4063544273e+00,
    8.8711702824e-01, -2.0000000000e+00, 1.6580896378e+00, -2.0000000000e+00,
    -2.0000000000e+00, -2.0000000000e+00, 4.6244382858e-01, -2.0000000000e+00,
    -2.0000000000e+00
};

const double DecisionTree::VALUE[NUM_NODES][NUM_CLASSES] = {
    {2.0280000000e+03, 2.0280000000e+03, 2.0280000000e+03, 2.0280000000e+03},
    {1.4570000000e+03, 9.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00},
    {1.4490000000e+03, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00},
    {8.0000000000e+00, 9.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00},
    {5.7100000000e+02, 2.0190000000e+03, 2.0280000000e+03, 2.0280000000e+03},
    {5.7100000000e+02, 1.9790000000e+03, 1.6790000000e+03, 5.8100000000e+02},
    {2.0000000000e+00, 4.9000000000e+01, 1.0490000000e+03, 0.0000000000e+00},
    {1.0000000000e+00, 1.9000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {1.0000000000e+00, 9.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 1.0000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {1.0000000000e+00, 3.0000000000e+01, 1.0490000000e+03, 0.0000000000e+00},
    {1.0000000000e+00, 6.0000000000e+00, 7.3600000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 4.7500000000e+02, 0.0000000000e+00},
    {1.0000000000e+00, 6.0000000000e+00, 2.6100000000e+02, 0.0000000000e+00},
    {1.0000000000e+00, 6.0000000000e+00, 1.3000000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 1.3100000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 2.4000000000e+01, 3.1300000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 2.4000000000e+01, 1.0900000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 9.0000000000e+00, 1.0900000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 1.5000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 2.0400000000e+02, 0.0000000000e+00},
    {5.6900000000e+02, 1.9300000000e+03, 6.3000000000e+02, 5.8100000000e+02},
    {4.6000000000e+02, 1.9000000000e+01, 0.0000000000e+00, 4.7400000000e+02},
    {1.4000000000e+01, 1.9000000000e+01, 0.0000000000e+00, 4.7400000000e+02},
    {1.4000000000e+01, 1.9000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {1.4000000000e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 1.9000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.7400000000e+02},
    {4.4600000000e+02, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00},
    {1.0900000000e+02, 1.9110000000e+03, 6.3000000000e+02, 1.0700000000e+02},
    {1.0600000000e+02, 1.6570000000e+03, 0.0000000000e+00, 1.0700000000e+02},
    {1.0600000000e+02, 1.6570000000e+03, 0.0000000000e+00, 0.0000000000e+00},
    {3.7000000000e+01, 1.8000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {3.0000000000e+00, 1.5000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {3.4000000000e+01, 3.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00},
    {6.9000000000e+01, 1.6390000000e+03, 0.0000000000e+00, 0.0000000000e+00},
    {6.8000000000e+01, 6.2900000000e+02, 0.0000000000e+00, 0.0000000000e+00},
    {1.0000000000e+00, 1.0100000000e+03, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.0700000000e+02},
    {3.0000000000e+00, 2.5400000000e+02, 6.3000000000e+02, 0.0000000000e+00},
    {3.0000000000e+00, 5.6000000000e+01, 6.3000000000e+02, 0.0000000000e+00},
    {2.0000000000e+00, 3.7000000000e+01, 6.3000000000e+02, 0.0000000000e+00},
    {2.0000000000e+00, 3.7000000000e+01, 1.1400000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 5.1600000000e+02, 0.0000000000e+00},
    {1.0000000000e+00, 1.9000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 1.9800000000e+02, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 4.0000000000e+01, 3.4900000000e+02, 1.4470000000e+03},
    {0.0000000000e+00, 4.0000000000e+01, 1.2100000000e+02, 1.3870000000e+03},
    {0.0000000000e+00, 4.0000000000e+01, 1.2100000000e+02, 5.4900000000e+02},
    {0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.4900000000e+02},
    {0.0000000000e+00, 4.0000000000e+01, 1.2100000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 4.0000000000e+01, 0.0000000000e+00, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 1.2100000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 8.3800000000e+02},
    {0.0000000000e+00, 0.0000000000e+00, 2.2800000000e+02, 6.0000000000e+01},
    {0.0000000000e+00, 0.0000000000e+00, 2.2800000000e+02, 0.0000000000e+00},
    {0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 6.0000000000e+01}
};

const int DecisionTree::CLASSES[NUM_CLASSES] = {
    1, 2, 3, 4
};

#endif // SUBCORE_CLASSIFIER_H

// Auto-generated sub-core classifier for C++ integration
// Generated from compact_classifier.py
// Model: Logistic Regression with 20 features

#ifndef SUBCORE_CLASSIFIER_H
#define SUBCORE_CLASSIFIER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

class SubcoreClassifier {
public:
    // Model configuration
    static constexpr int NUM_FEATURES = 20;
    static constexpr int NUM_CLASSES = 4;
    
    // Feature names (for reference)
    static const std::vector<std::string> FEATURE_NAMES;
    
    // Model parameters
    static const double SCALER_MEAN[NUM_FEATURES];
    static const double SCALER_STD[NUM_FEATURES];
    static const double COEF[NUM_CLASSES][NUM_FEATURES];
    static const double INTERCEPT[NUM_CLASSES];
    static const int CLASSES[NUM_CLASSES];
    
    // Predict optimal number of sub-cores
    static int predict(const double* features) {
        // Standardize features
        double scaled_features[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            scaled_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_STD[i];
        }
        
        // Compute logits for each class
        double max_logit = -1e9;
        int best_class = CLASSES[0];
        
        for (int c = 0; c < NUM_CLASSES; c++) {
            double logit = INTERCEPT[c];
            for (int f = 0; f < NUM_FEATURES; f++) {
                logit += COEF[c][f] * scaled_features[f];
            }
            
            if (logit > max_logit) {
                max_logit = logit;
                best_class = CLASSES[c];
            }
        }
        
        return best_class;
    }
    
    // Predict with probabilities
    static int predict_proba(const double* features, double* probabilities) {
        // Standardize features
        double scaled_features[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            scaled_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_STD[i];
        }
        
        // Compute logits
        double logits[NUM_CLASSES];
        double max_logit = -1e9;
        for (int c = 0; c < NUM_CLASSES; c++) {
            logits[c] = INTERCEPT[c];
            for (int f = 0; f < NUM_FEATURES; f++) {
                logits[c] += COEF[c][f] * scaled_features[f];
            }
            if (logits[c] > max_logit) {
                max_logit = logits[c];
            }
        }
        
        // Softmax
        double sum_exp = 0.0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            probabilities[c] = std::exp(logits[c] - max_logit);
            sum_exp += probabilities[c];
        }
        for (int c = 0; c < NUM_CLASSES; c++) {
            probabilities[c] /= sum_exp;
        }
        
        // Return class with max probability
        int best_class = CLASSES[0];
        double max_prob = probabilities[0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (probabilities[c] > max_prob) {
                max_prob = probabilities[c];
                best_class = CLASSES[c];
            }
        }
        
        return best_class;
    }
};

// Feature names
const std::vector<std::string> SubcoreClassifier::FEATURE_NAMES = {
    "gpu_sim_insn",
    "gpu_occupancy",
    "gpu_ipc",
    "avg_FP_INT",
    "avg_SCHEDP",
    "avg_TOT_INST",
    "avg_PIPEP",
    "avg_PIPE_A",
    "avg_IBP",
    "avg_DC_RM",
    "avg_INT_ACC",
    "avg_INTP",
    "avg_RFP",
    "gpu_sim_cycle",
    "avg_REG_RD",
    "avg_INT_MULP",
    "avg_INT_MUL_ACC",
    "avg_REG_WR",
    "avg_IDLE_CORE_N",
    "avg_IDLE_COREP"
};

const double SubcoreClassifier::SCALER_MEAN[NUM_FEATURES] = {
    1.4245597101e+02, 1.3646151186e+01, 1.2266468434e+07, 1.6935339469e+04,
    3.5408492299e-02, 1.0641120328e+00, 3.4213991660e-01, 1.2270503107e-01,
    9.5052613711e-02, 1.2911372561e+00, 1.9917180153e+01, 2.6605708508e+03,
    2.4253585054e+03, 2.2100807312e+02, 1.2172841035e+05, 7.7535929880e+04,
    3.2719461410e+05, 1.4014179498e+03, 1.1397606808e-01, 7.0430579359e+01
};

const double SubcoreClassifier::SCALER_STD[NUM_FEATURES] = {
    4.4453755148e+02, 1.4248328387e+01, 1.0693057748e+08, 6.1662121863e+04,
    1.0094476878e-01, 3.2100952011e+00, 1.2112574820e+00, 4.1958186257e-01,
    2.7648053488e-01, 1.1494452043e+00, 4.9280240487e+00, 7.5849230455e+03,
    7.0546666042e+03, 1.4786884419e+03, 3.9064771140e+05, 2.1076459404e+05,
    1.1583471151e+06, 4.7920579596e+03, 1.0146809242e-01, 1.7426341652e+01
};

const double SubcoreClassifier::COEF[NUM_CLASSES][NUM_FEATURES] = {
    {-1.9233466378e+00, -8.8399383224e-01, -1.1222596290e+00, 1.1255211224e-01,
     5.4301190395e-01, 4.6471672155e-01, 1.8578950471e+00, -2.6238128007e+00,
     -1.6979632042e+00, 4.8339771703e-01, -3.4798617558e+00, 5.4302917328e-01,
     -1.6979491349e+00, 1.4325330009e-01, -3.0726717932e-01, 2.1295167458e+00,
     1.8579167353e+00, -2.6238127830e+00, 4.8341514813e-01, -3.4799104311e+00
    },
    {1.9293368628e+00, -1.4094938895e+00, 1.7195258151e-01, -8.4474027129e-01,
     -4.8058929193e-01, -7.1269162492e-01, -1.3300592539e+00, 1.4028496489e+00,
     3.5595868138e-01, 5.9069346289e-01, -1.3339248559e+00, -4.8060374256e-01,
     3.5593189016e-01, -4.1776228677e-01, -2.3036385167e+00, 2.7520710531e+00,
     -1.3300496943e+00, 1.4028514372e+00, 5.9061544946e-01, -1.3337678063e+00
    },
    {-1.3058384751e+00, 2.3766991436e-01, 4.2504070650e-01, 4.0118834632e-01,
     -6.1332546733e-01, -3.8734861359e-01, -4.4404046514e-01, 2.6585367128e+00,
     5.1002942364e-01, 5.4852853988e-01, 8.0071556964e-01, -6.1332741397e-01,
     5.1004159839e-01, 1.5129645255e+00, 1.4966648282e+00, -4.4654023457e+00,
     -4.4407296859e-01, 2.6585347566e+00, 5.4859083581e-01, 8.0066658159e-01
    },
    {1.2998482500e+00, 2.0558178073e+00, 5.2526634097e-01, 3.3099981273e-01,
     5.5090285531e-01, 6.3532351696e-01, -8.3795328071e-02, -1.4375735610e+00,
     8.3197509919e-01, -1.6226197198e+00, 4.0130710420e+00, 5.5090198325e-01,
     8.3197564634e-01, -1.2384555388e+00, 1.1142408678e+00, -4.1618545320e-01,
     -8.3794072321e-02, -1.4375734108e+00, -1.6226214334e+00, 4.0130116557e+00
    }
};

const double SubcoreClassifier::INTERCEPT[NUM_CLASSES] = {
    2.0810538793e+00, 1.9532083220e+00, 1.1258952175e+00, -5.1601574187e+00
};

const int SubcoreClassifier::CLASSES[NUM_CLASSES] = {
    1, 2, 3, 4
};

#endif // SUBCORE_CLASSIFIER_H

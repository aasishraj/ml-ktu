#include <stddef.h>
#include <stdio.h>
#include <math.h>

#define DATASET_LEN 5
#define INPUT_LEN 3 // input vector
#define CLASSES 3 // no. of classes

float calculate_posterior_probability (
    float dataset_x[DATASET_LEN][INPUT_LEN],
    float dataset_y[DATASET_LEN],
    float input[INPUT_LEN]
) {
    // Bernoulli Naive Bayes with Laplace smoothing (alpha = 1)
    size_t num_samples = DATASET_LEN;
    size_t num_features = INPUT_LEN;

    // Count class occurrences
    size_t count_y1 = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        if (dataset_y[i] >= 0.5f) {
            count_y1++;
        }
    }
    size_t count_y0 = num_samples - count_y1;

    // Priors with Laplace smoothing
    float prior1 = (count_y1 + 1.0f) / (num_samples + 2.0f);
    float prior0 = (count_y0 + 1.0f) / (num_samples + 2.0f);

    float log_p1 = logf(prior1);
    float log_p0 = logf(prior0);

    // Likelihoods per feature
    for (size_t f = 0; f < num_features; ++f) {
        size_t count_x1_y1 = 0;
        size_t count_x1_y0 = 0;

        for (size_t i = 0; i < num_samples; ++i) {
            if (dataset_x[i][f] >= 0.5f) {
                if (dataset_y[i] >= 0.5f) {
                    count_x1_y1++;
                } else {
                    count_x1_y0++;
                }
            }
        }

        // P(x_f = 1 | y) with Laplace smoothing
        float p_x1_given_y1 = (count_x1_y1 + 1.0f) / (count_y1 + 2.0f);
        float p_x0_given_y1 = 1.0f - p_x1_given_y1;
        float p_x1_given_y0 = (count_x1_y0 + 1.0f) / (count_y0 + 2.0f);
        float p_x0_given_y0 = 1.0f - p_x1_given_y0;

        if (input[f] >= 0.5f) {
            log_p1 += logf(p_x1_given_y1);
            log_p0 += logf(p_x1_given_y0);
        } else {
            log_p1 += logf(p_x0_given_y1);
            log_p0 += logf(p_x0_given_y0);
        }
    }

    float p1 = expf(log_p1);
    float p0 = expf(log_p0);
    float denom = p1 + p0;
    if (denom == 0.0f) {
        return 0.5f;
    }
    return p1 / denom;
}

int main () {

    // sample data
    float dataset_x[DATASET_LEN][INPUT_LEN] = {
        {1.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 0.0,0.0}
    };

    float dataset_y[DATASET_LEN] = {
        1.0,
        0.0,
        1.0,
        0.0,
        1.0
    };

    float input[INPUT_LEN] = {1.0, 1.0, 0.0};
    float posterior_y1 = calculate_posterior_probability(dataset_x, dataset_y, input);
    int predicted = posterior_y1 >= 0.5f ? 1 : 0;

    printf("P(y=1 | x) = %.6f\n", posterior_y1);
    printf("Predicted class: %d\n", predicted);

    return 0;
}
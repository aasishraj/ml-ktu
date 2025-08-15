#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WEIGHTS_COUNT 3
#define DATASET_LEN 10
#define EPOCHS 10000000
#define MAX_RAND_FLOAT 100.0f
#define STEP_SIZE 0.00001f

float random_float_generator(void) {
    // Uniform in [0, MAX_RAND_FLOAT)
    return ((float)rand() / (float)RAND_MAX) * MAX_RAND_FLOAT;
}

void forward_pass(float y_pred[DATASET_LEN],
                  const float weights[WEIGHTS_COUNT],
                  float bias,
                  const float inputs[DATASET_LEN][WEIGHTS_COUNT]) {
    for (size_t i = 0; i < DATASET_LEN; i++) {
        float acc = bias;
        for (size_t j = 0; j < WEIGHTS_COUNT; j++) {
            acc += weights[j] * inputs[i][j];
        }
        y_pred[i] = acc;
    }
}

void gradient_descent(float gradient_weights[WEIGHTS_COUNT],
                      float *gradient_bias,
                      const float y_pred[DATASET_LEN],
                      const float outputs[DATASET_LEN],
                      const float inputs[DATASET_LEN][WEIGHTS_COUNT]) {
    // d/dw_i of MSE = 2/N * sum( (y_pred - y_true) * x_i )
    for (size_t i = 0; i < WEIGHTS_COUNT; i++) {
        float gw = 0.0f;
        for (size_t j = 0; j < DATASET_LEN; j++) {
            gw += (y_pred[j] - outputs[j]) * inputs[j][i];
        }
        gradient_weights[i] = (2.0f / DATASET_LEN) * gw;
    }

    // d/db of MSE = 2/N * sum( y_pred - y_true )
    float gb = 0.0f;
    for (size_t i = 0; i < DATASET_LEN; i++) {
        gb += (y_pred[i] - outputs[i]);
    }
    *gradient_bias = (2.0f / DATASET_LEN) * gb;
}

int main(void) {
    // Optional: seed for reproducibility control
    srand((unsigned)time(NULL));

    float inputs[DATASET_LEN][WEIGHTS_COUNT] = {
        {-2.509f,  9.014f,  4.640f},
        { 1.973f, -6.880f, -6.880f},
        {-8.838f,  7.324f,  2.022f},
        { 4.161f, -9.588f,  9.398f},
        { 6.649f, -5.753f, -6.364f},
        {-6.332f, -3.915f,  0.495f},
        {-1.361f, -4.175f,  2.237f},
        {-7.210f, -4.157f, -2.673f},
        {-0.879f,  5.704f, -6.007f},
        { 0.285f,  1.848f, -9.071f}
    };

    float outputs[DATASET_LEN] = {
        -14.572f, 14.812f, -29.326f, 38.870f, 24.074f,
        -2.876f, 9.695f, -10.421f, -12.476f, -7.815f
    };

    // Model init
    float weights[WEIGHTS_COUNT];
    float bias;

    for (size_t i = 0; i < WEIGHTS_COUNT; i++)
        weights[i] = random_float_generator() - 0.5f; // small symmetric init

    bias = random_float_generator() - 0.5f;

    float y_pred[DATASET_LEN];
    float gradient_weights[WEIGHTS_COUNT];
    float gradient_bias = 0.0f;

    // Initial forward and gradient
    forward_pass(y_pred, weights, bias, inputs);
    gradient_descent(gradient_weights, &gradient_bias, y_pred, outputs, inputs);

    for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
        printf("\nEPOCH: %zu\n", epoch);

        // Parameter update
        for (size_t j = 0; j < WEIGHTS_COUNT; j++) {
            weights[j] -= STEP_SIZE * gradient_weights[j];
            printf("W%zu: %f\n", j, weights[j]);
        }
        bias -= STEP_SIZE * gradient_bias;
        printf("B: %f\n", bias);

        // Recompute predictions and gradients
        forward_pass(y_pred, weights, bias, inputs);
        gradient_descent(gradient_weights, &gradient_bias, y_pred, outputs, inputs);
    }

    return 0;
}

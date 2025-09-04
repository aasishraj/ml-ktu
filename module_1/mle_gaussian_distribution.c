#include <stdio.h>
#include <math.h>


double calculate_mle_mean(double dataset[], int dataset_len) {
    double sum = 0;
    for (int i = 0; i < dataset_len; i++)
        sum += dataset[i];

    double mean = sum / dataset_len;
    return mean;
}

double calculate_mle_variance(double dataset[], int dataset_len, double mean) {
    double sum = 0;
    for (int i = 0; i < dataset_len; i++)
        sum += (dataset[i] - mean) * (dataset[i] - mean);

    double var = sum / dataset_len;
    return var;
}

double gaussian_density(double x, double mean, double var) {
    double pdf = exp(-((x - mean) * (x - mean)) / (2 * var)) / sqrt(2 * M_PI * var);

    return pdf;
}

int main(int argc, char **argv) {
    double dataset[] = {5.0, 5.1, 5.2, 4.9, 5.0};
    int dataset_len = sizeof(dataset) / sizeof(double);

    double mean = calculate_mle_mean(dataset, dataset_len);
    double var = calculate_mle_variance(dataset, dataset_len, mean);

    printf("MLE Mean: %lf\n",mean);
    printf("MLE Variance: %lf\n", var);

    printf("\nEnter Value to Predict: ");
    double x;
    scanf("%lf", &x);

    double pdf = gaussian_density(x, mean, var);
    printf("Probability Density: %lf\n", pdf);

    return 0;
}
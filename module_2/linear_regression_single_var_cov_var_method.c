#include <stdio.h>

#define DATA_LEN 10

int main(int argc, char const *argv[]) {
  double dataset[DATA_LEN][2] = {{500, 150000},  {750, 200000},  {1000, 250000},
                                 {1250, 300000}, {1500, 350000}, {1750, 400000},
                                 {2000, 450000}, {2250, 500000}, {2500, 550000},
                                 {2750, 600000}};

  float x_mean = 0, y_mean = 0;

  for (int i = 0; i < DATA_LEN; i++) {
    x_mean += dataset[i][0];
    y_mean += dataset[i][1];
  }

  // Calculate the mean of x and y
  x_mean /= DATA_LEN;
  y_mean /= DATA_LEN;

  // Calculate the covariance and variance
  float cov = 0, var = 0;

  for (int i = 0; i < DATA_LEN; i++) {
    float x_diff = x_mean - dataset[i][0];
    float y_diff = y_mean - dataset[i][1];

    cov += x_diff * y_diff;
    var += x_diff * x_diff;
  }

  cov /= DATA_LEN - 1;
  var /= DATA_LEN - 1;

  // Calculate the slope and intercept
  float m = 0, b = 0;

  // y = mx + b
  m = cov / var;
  b = y_mean - m * x_mean;

  float x_val, y_val;

  printf("y = %.4f * x + %.4f\n", m, b);

  printf("Enter the value of x: ");
  scanf("%f", &x_val);
  y_val = m * x_val + b;
  printf("The value of y is: %.4f\n", y_val);

  return 0;
}
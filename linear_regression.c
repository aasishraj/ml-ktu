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

  x_mean /= DATA_LEN;
  y_mean /= DATA_LEN;

  printf("Mean\nX: %f, Y: %f\n\n", x_mean, y_mean);

  float cov = 0, var = 0;

  for (int i = 0; i < DATA_LEN; i++) {
    float x_diff = x_mean - dataset[i][0];
    float y_diff = y_mean - dataset[i][1];

    cov += x_diff * y_diff;
    var += x_diff * x_diff;
  }

  cov /= DATA_LEN - 1;
  var /= DATA_LEN - 1;

  printf("var: %f, cov: %f\n\n", var, cov);

  // y = mx + b
  float m = 0, b = 0;

  m = cov / var;
  b = y_mean - m * x_mean;

  printf("m: %f, b: %f\n\n", m, b);

  float x_val, y_val;

  printf("X: ");
  scanf("%f", &x_val);
  y_val = m * x_val + b;
  printf("Y: %.4f\n", y_val);

  return 0;
}
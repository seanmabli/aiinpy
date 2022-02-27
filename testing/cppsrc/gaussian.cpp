#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

class gaussian {
  public:
    vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return exp(-pow(input, 2)); });
      return input;
    }

    vector <double> backward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return -2 * input * exp(-pow(input, 2)); });
      return input;
    }
};

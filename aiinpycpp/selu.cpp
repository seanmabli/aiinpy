#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

class selu {
  public:
    vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return 1.0507 * ((input < 0) ? (1.67326 * (exp(input) - 1)) : input); });
      return input;
    }

    vector <double> backward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return 1.0507 * ((input < 0) ? (1.67326 * exp(input)) : input); });
      return input;
    }
};

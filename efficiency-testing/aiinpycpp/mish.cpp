#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

class relu {
  public: 
    vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return (input * ((2 * exp(input)) + exp(2 * input))) / ((2 * exp(input)) + exp(2 * input) + 2); });
      return input;
    }

    vector <double> backward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return (exp(input) * ((4 * exp(2 * input)) + exp(3 * input) + (4 * (1 + input)) + (exp(input) * (6 + (4 * input))))) / pow(2 + (2 * exp(input)) + exp(2 * input), 2); });
      return input;
    }
};

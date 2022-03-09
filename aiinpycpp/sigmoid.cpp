#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

class sigmoid {
  public:
    vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return 1 / (1 + exp(-input)); });
      return input;
    }

    vector <double> backward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return input * (1 - input); });
      return input;
    }
};

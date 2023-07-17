#include <iostream>
#include <vector>
#include <algorithm>

class leakyrelu {
  public:
    double _alpha;
 
    leakyrelu(double alpha = 0.01) {
      _alpha = alpha;
    }

    vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [&](double input){ return (_alpha * input < input) ? input : 0; });
      return input;
    }

    vector <double> backward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [&](double input){ return (input <= 0) ? _alpha : 1; });
      return input;
    }
};

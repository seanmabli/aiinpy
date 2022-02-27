#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class leakyrelu {
  public:
    double a;
 
    leakyrelu(double alpha = 0.01) {
      a = alpha;
    }

    vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [&](double input){ return (a * input < input) ? input : 0; });
      return input;
    }

    vector <double> backward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [&](double input){ return (input <= 0) ? a : 1; });
      return input;
    }
};

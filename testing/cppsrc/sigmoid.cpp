#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "printvector.cpp"

using namespace std;

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

int main() {
  vector <double> input = { 1.5483257342, 3.54324, 4.54832905, 5.87629 };

  sigmoid activation;

  printvector("forward: ", activation.forward(input));
  printvector("backward: ", activation.backward(input));  

  return 0;
}

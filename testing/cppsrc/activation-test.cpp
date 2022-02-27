#include <iostream>
#include <vector>
#include <algorithm>

#include "printvector.cpp"
#include "binarystep.cpp"
#include "sigmoid.cpp"
#include "gaussian.cpp"
#include "identity.cpp"
#include "relu.cpp"
#include "selu.cpp"

using namespace std;

int main() {
  vector <double> input = { -1.5483257342, 3.54324, -4.54832905, 5.87629 };

  selu activation;

  printvector("forward: ", activation.forward(input));
  printvector("backward: ", activation.backward(input));  

  return 0;
}

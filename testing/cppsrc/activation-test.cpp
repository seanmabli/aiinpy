#include <iostream>
#include <array>
#include <vector>
#include <algorithm>

#include "print.cpp"
// #include "binarystep.cpp"
// #include "sigmoid.cpp"
// #include "gaussian.cpp"
// #include "identity.cpp"
// #include "relu.cpp"
// #include "selu.cpp"
// #include "leakyrelu.cpp"

using namespace std;

int main() {
  double input[] = { -1.5483257342, 3.54324, -4.54832905, 5.87629 };

  // binarystep activation;

  transform(begin(input), end(input), begin(input), [](double input){ return input * input; });

  printarraydouble("forward: ", input);

  // printvectordouble("forward: ", activation.forward(input));
  // printvectordouble("backward: ", activation.backward(input));  

  return 0;
}

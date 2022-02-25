#include <iostream>
#include <cmath>

using namespace std;
class sigmoid {
  public:

  double forward(double input) {
    return 1 / (1 + exp(-input));
  }

  double backward(double input) {
    return input * (1 - input);
  }
};

int main()
{
  sigmoid activation;
  double input = 0.3;
  double outputforward = activation.forward(input);
  double outputbackward = activation.backward(input);

  cout "forward: " << outputforward << "\n";
  cout "backward: "  << outputbackward < "\n";

  return 0;
}

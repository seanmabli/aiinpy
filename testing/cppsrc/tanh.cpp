#include <iostream>
#include <cmath>

using namespace std;
class tanha {
  public:

  double forward(double input) {
    return (exp(input) - exp(-input)) / (exp(input) + exp(-input));
  }

  double backward(double input) {
    return 1 - pow(input, 2);
  }
};

int main()
{
  tanha activation;
  double input = 0.3;
  double outputforward = activation.forward(input);
  double outputbackward = activation.backward(input);

  cout << outputforward << outputbackward;

  return 0;
}

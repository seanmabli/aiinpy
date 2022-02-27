#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class binarystep {
	public:	
		vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return (input < 0) ? 0 : 1; });
      return input;
		}

		vector <double> backward(vector <double> input) {
			return { [1] * input.size() };
		}
};



int main() {
	binarystep activation;
	double input = 0.3;
	double outputforward = activation.forward(input);
	double outputbackward = activation.backward(input);

	cout << "forward: " << outputforward << "\n";
	cout << "backward: " << outputbackward << "\n";

	return 0;
}

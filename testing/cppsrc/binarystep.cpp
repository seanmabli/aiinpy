#include <iostream>

using namespace std;
class binarystep {
	public:
		
		double forward(double input) {
			return (input < 0) ? 0 : 1;
		}

		double backward(double input) {
			return 1;
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

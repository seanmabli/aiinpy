#include <iostream>
#include <vector>
#include <algorithm>

class binarystep {
	public:	
		vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [](double input){ return (input < 0) ? 0 : 1; });
      return input;
		}

		vector <double> backward(vector <double> input) {
			return vector <double> (input.size(), 1);
		}
};

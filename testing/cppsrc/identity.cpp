#include <iostream>
#include <vector>
#include <algorithm>

class identity {
  public: 
    vector <double> forward(vector <double> input) {
      return input;
    }

    vector <double> backward(vector <double> input) {
      return vector <double> (input.size(), 1);
    }
};

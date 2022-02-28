#include <iostream>
#include <vector>
#include <algorithm>

#include "print.cpp"

class nn {
  public:
    vector <int> _inshape;
    vector <int> _outshape;
    int _learningrate;
    vector <int> _weightsinit;
    vector <int> _biasesinit;

    nn(vector <int> inshape, vector <int> outshape, int learningrate, vector <int> weightsinit = { -1, 1 }, vector <int> biasesinit = { 0, 0 }) {
      _inshape = inshape;
      _outshape = outshape;
      _learningrate = learningrate;
      _weightsinit = weightsinit;
      _biasesinit = biasesinit;
      
      vector <vector <double>> weights = 


    }
/*
    vector <double> forward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [&](double input){ return (a * input < input) ? input : 0; });
      return input;
    }

    vector <double> backward(vector <double> input) {
      transform(input.begin(), input.end(), input.begin(), [&](double input){ return (input <= 0) ? a : 1; });
      return input;
    }
*/
};

int main() {
  nn model({ 1, 6, 82 }, { 53, 6, 245 });
}

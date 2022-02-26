#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

double sigmoidforward(double input) { return 1 / (1 + exp(-input)); }
double sigmoidbackward(double input) { return input * (1 - input); }

int main() {
  vector <double> input = { 1.5483257342, 3.54324, 4.54832905, 5.87629 };

  vector <double> outputforward (input.size());
  vector <double> outputbackward (input.size());

  transform(input.begin(), input.end(), outputforward.begin(), sigmoidforward);
  cout << "forward: ";
  for (int i = 0; i < outputforward.size(); i++)
		cout << outputforward[i] << ' ';
  cout << "\n";

  transform(input.begin(), input.end(), outputbackward.begin(), sigmoidbackward);
  cout << "backward: ";
  for (int i = 0; i < outputbackward.size(); i++)
		cout << outputbackward[i] << ' ';
  cout << "\n";

  return 0;
}
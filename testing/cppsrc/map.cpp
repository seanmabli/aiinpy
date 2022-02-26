#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void printVector(vector<double> v) {
	for (int i = 0; i < v.size(); i++)
		cout << v[i] << ' ';
	cout << endl;
}

int main() {
  vector <double> input = { 1.5483257342, 3.54324, 4.54832905, 5.87629 };
  
  vector <double> output;
  output.resize(input.size());
	
  double scalar = 4.57342895;
	printVector(input);

  transform(input.begin(), input.end(), output.begin(), [scalar](double c){ return c * scalar; });
  printVector(input);
	printVector(output);
}
using namespace std;

void printvectordouble(const char label[], vector <double> input) {
  cout << label;
  for (int i = 0; i < input.size(); i++)
    cout << input[i] << ' ';
  cout << "\n";
}

void printvectorint(const char label[], vector <int> input) {
  cout << label;
  for (int i = 0; i < input.size(); i++)
    cout << input[i] << ' ';
  cout << "\n";
}

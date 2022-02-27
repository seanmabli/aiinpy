using namespace std;

void printvector(char label[], vector <double> input) {
  cout << label;
  for (int i = 0; i < input.size(); i++)
    cout << input[i] << ' ';
  cout << "\n";
}

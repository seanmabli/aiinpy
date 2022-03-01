/*
#include <iostream>
#include <chrono>

using namespace std;

int main()
{
  int dec = chrono::system_clock::to_time_t(chrono::system_clock::now());
  cout << dec << " ";

  string bin{};
    
  while(dec > 0) {
    if(dec % 2 == 0) { bin.insert(bin.begin( ), '0'); } else { bin.insert(bin.begin( ), '1'); }     
    dec >>= 1;
  }
  
  cout << bin << "\n";
}
*/

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> count(6);
  
    random_device rnd_device;
    mt19937 mersenne_engine {rnd_device()};
    uniform_int_distribution<int> dist {0, 5};

    vector<int> vec(1000000000);
    generate(vec.begin(), vec.end(), [&dist, &mersenne_engine](){ return dist(mersenne_engine); });
    
    for (auto i : vec) {
      count[i] += 1;
    }
  
    for (auto i : count) {
      cout << i << ' ';
    }
}
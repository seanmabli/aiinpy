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

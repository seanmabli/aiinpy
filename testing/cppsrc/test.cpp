#include <iostream>
#include <vector>
 
int main()
{
    int val = 1;
    unsigned int n = 5;
 
    // fill constructor
    std::vector<int> vec(n, val);
 
    for (int i: vec) {
        std::cout << i << ' ';
    }
 
    return 0;
}

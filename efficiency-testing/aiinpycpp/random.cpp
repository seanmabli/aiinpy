
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

template<typename T>
double getAverage(std::vector<T> const& v) {
    if (v.empty()) {
        return 0;
    }
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}

int main() {
    random_device rnd_device;
    mt19937 mersenne_engine {rnd_device()};
    uniform_real_distribution<double> dist {0, 1};

    vector<vector<double>> vec(10);
    generate(vec.begin(), vec.end(), [&dist, &mersenne_engine](){ return dist(mersenne_engine); });
    
    // double avg = getAverage(vec);
    // cout << "Average is " << avg << '\n';
}

/*
#include <random>
#include <iostream>
using namespace std;
int main()
{
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(0,1.0);
    for (int i = 0; i < 5; ++i) {
            cout << dis(gen) << '\n';
    }
}
*/
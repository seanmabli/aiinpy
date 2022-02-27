#include <iostream>
 
using namespace std;

class box {
   public:
      box(double l = 2.0, double b = 2.0, double h = 2.0) {
         length = l;
         breadth = b;
         height = h;
      }
      double Volume() {
         return length * breadth * height;
      }

      double length;
      double breadth;
      double height;
};

int main() {
   box Box1(3.3, 1.2, 1.5);
   box Box2(8.5, 6.0, 2.0);

   cout << Box1.Volume() << "\n";
   cout << Box2.Volume() << "\n";

   return 0;
}

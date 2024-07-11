#include "../comp.h"
using namespace std;

int main() {
  // solution
  vi arr;
  int x;
  while (cin >> x) arr.PB(x);
  int index = 0;
  for (int b = arr.size() / 2; b > 0; b /= 2) {
    if (index == arr.size() - 1) break;
    while (arr[index + b] < arr[index + b + 1]) index += b;
  }
  index++;
  cout << index << "\n";
}

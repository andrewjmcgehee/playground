#include "../comp.h"
using namespace std;

int main() {
  // solution
  vi arr;
  int x;
  while (cin >> x) {
    arr.PB(x);
  }
  int target = arr[arr.size() - 1];
  arr.pop_back();
  int index = 0;
  for (int b = arr.size() / 2; b >= 1; b /= 2) {
    while (index + b < arr.size() && arr[index + b] <= target) index += b;
  }
  if (arr[index] != target) index = -1;
  cout << index << "\n";
}

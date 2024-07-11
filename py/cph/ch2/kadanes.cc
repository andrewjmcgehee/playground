#include "../comp.h"
using namespace std;

int main() {
  vi arr;
  int x;
  while (cin >> x) {
    arr.PB(x);
  }
  int best = 0;
  int sum = 0;
  FOR(i, 0, arr.size()) {
    sum = max(arr[i], sum + arr[i]);
    best = max(sum, best);
  }
  cout << best << "\n";
  return 0;
}

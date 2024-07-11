#include <algorithm>
#include <any>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <ccomplex>
#include <cctype>
#include <cerrno>
#include <cfenv>
#include <cfloat>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <ciso646>
#include <climits>
#include <clocale>
#include <cmath>
#include <codecvt>
#include <complex>
#include <condition_variable>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctgmath>
#include <ctime>
#include <cwchar>
#include <cwctype>
#include <deque>
#include <exception>
#include <filesystem>
#include <forward_list>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <random>
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <variant>
#include <vector>

typedef long long ll;
typedef std::vector<int> vi;
typedef std::vector<long long> vl;
typedef std::vector<double> vd;
typedef std::vector<std::string> vs;
typedef std::vector<std::vector<int>> vvi;
typedef std::vector<std::vector<long long>> vvl;
typedef std::vector<std::vector<double>> vvd;
typedef std::vector<std::vector<std::string>> vvs;
typedef std::pair<int, int> pi;
typedef std::pair<long long, long long> pl;
typedef std::pair<double, double> pd;

#define F first
#define S second
#define PB push_back
#define MP make_pair
#define FOR(i, a, b) for (int i = a; i < b; i++)
#define RANGE(i, n) for (int i = 0; i < n; i++)
#define DFOR(i, a, b) for (int i = a; i > b; i--)
#define DRANGE(i, n) for (int i = n; i > 0; i--)
#define ALL(v) v.begin(), v.end()
#define SQ(a) (a) * (a)

const double PI = std::acos(-1.0);

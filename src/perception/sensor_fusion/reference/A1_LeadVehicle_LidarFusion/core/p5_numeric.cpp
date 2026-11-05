// Offline numeric regression executable, not a vehicle-control node.
// Implements only the reference default config and constant ego-speed plan.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

double number(const char* s) {
  std::size_t end = 0;
  double v = std::stod(s, &end);
  if (end != std::string(s).size() || !std::isfinite(v))
    throw std::invalid_argument("finite numeric argument required");
  return v;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: p5_numeric ego_mps gap_m lead_mps\n";
    return 2;
  }
  try {
    const double ego = number(argv[1]), gap = number(argv[2]);
    const double lead = std::max(0.0, number(argv[3]));
    if (ego < 0) throw std::invalid_argument("ego speed must be nonnegative");
    constexpr int n = 40, horizon = 10;
    constexpr double dt = 0.1;
    const double min_gap = 7.0 + 6.0 * std::log1p(ego / 5.0);
    std::array<double, n> t{}, u{};
    double x = 0;
    for (int i = 0; i < n; ++i) {
      t[i] = i * dt;
      if (i) x += ego * dt;
      u[i] = std::min(ego, std::max(0.0, (gap + lead * t[i] - x - min_gap) / 2.0));
    }
    for (int i = n - 2; i >= 0; --i)
      u[i] = std::min(u[i], u[i+1] + 20.0 * (t[i+1] - t[i]));
    u[0] = ego;
    for (int i = 1; i < n; ++i) {
      const double step = t[i] - t[i-1];
      u[i] = std::max(0.0, std::clamp(u[i], u[i-1] - 20.0 * step, u[i-1] + 5.0 * step));
    }
    double a = (u[horizon] - u[0]) / t[horizon];
    if (std::abs(a) < 1e-6) a = 0;
    std::cout << std::setprecision(17) << a << ',' << std::clamp(a, -3.0, 0.0) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}

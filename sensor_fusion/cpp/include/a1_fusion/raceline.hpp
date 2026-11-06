#pragma once
#include <Eigen/Dense>
#include <deque>
#include <string>
#include <utility>
#include <vector>
namespace a1::inspection {
// Reference path in map coordinates. Used here for target selection and logging only.
class Raceline {
 public:
  struct Frenet { double s{},n{},yaw{}; };  // n > 0 left of the line.
  static Raceline load(const std::string& csv_path,bool closed_loop);
  double length() const { return length_; }
  // hint < 0 searches the whole line; otherwise window points around the previous index.
  Frenet project(const Eigen::Vector2d& xy,int& hint,int window=101) const;
  // Signed distance along the line; on a loop it takes the shorter way round.
  double ahead(double s_from,double s_to) const;
 private:
  struct Point { double s,x,y; };
  std::vector<Point> points_;
  double length_{};
  bool closed_{};
};
// Lanes are judged against the ego's own offset, not the absolute racing line.
bool same_lane(double target_n,double ego_n,bool retained);
// A wrong file or origin shows up as the ego sitting far from the line for a while.
class AlignmentMonitor {
 public:
  void add(double stamp,double ego_abs_n);
  bool misaligned() const;
 private:
  std::deque<std::pair<double,double>> samples_;
};
}

#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>
namespace a1::inspection {
// Median LiDAR scan rate. Track confirmation and camera evidence count scans, not seconds, and are tuned for
// 10 Hz: the evidence cap 0.225/(1-exp(-1/(4f))) is 9.1 at 10 Hz and 3.7 at 4 Hz, and stays under the 3.0
// confirm level below about 3.2 Hz. The median ignores a single late scan.
class ScanRate {
 public:
  enum class State { Unknown, Ok, Slow, TooSlow };
  ScanRate()=default;
  ScanRate(double warn_hz,double min_hz):warn_hz_(warn_hz),min_hz_(min_hz){
    if(!std::isfinite(warn_hz)||!std::isfinite(min_hz)||min_hz<=0||warn_hz<min_hz)throw std::invalid_argument("scan rate limits need 0 < min <= warn");
  }
  void add(double stamp){
    if(!std::isfinite(stamp))return;
    if(have_&&stamp<=last_)intervals_.clear();  // time repeated or went backwards: measure again
    else if(have_){intervals_.push_back(stamp-last_);if(intervals_.size()>kWindow)intervals_.pop_front();}
    last_=stamp;have_=true;
  }
  // 0 until kMinSamples intervals are known.
  double hz() const {
    if(intervals_.size()<kMinSamples)return 0.;
    std::vector<double> sorted(intervals_.begin(),intervals_.end());std::sort(sorted.begin(),sorted.end());
    return 1./sorted[sorted.size()/2];
  }
  State state() const {
    if(intervals_.size()<kMinSamples)return State::Unknown;
    const double rate=hz();
    return rate<min_hz_?State::TooSlow:rate<warn_hz_?State::Slow:State::Ok;
  }
  double last_interval() const { return intervals_.empty()?std::numeric_limits<double>::quiet_NaN():intervals_.back(); }
  double warn_hz() const { return warn_hz_; }
  double min_hz() const { return min_hz_; }
 private:
  static constexpr size_t kWindow=11,kMinSamples=5;
  double warn_hz_{7.},min_hz_{4.},last_{};
  bool have_{};
  std::deque<double> intervals_;
};
}

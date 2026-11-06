#include "a1_fusion/scan_rate.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
using a1::inspection::ScanRate;
using State=ScanRate::State;
namespace {
// Feeds `count` scans spaced `period` apart, starting after the last stamp in `t`.
void feed(ScanRate& rate,double& t,int count,double period){for(int i=0;i<count;++i){t+=period;rate.add(t);}}
}
int main(){try{
  ScanRate rate(7.,4.);double t=100.;
  CHECK(rate.state()==State::Unknown&&rate.hz()==0.);
  rate.add(t);feed(rate,t,4,.1);                                     // 4 intervals are not enough yet
  CHECK(rate.state()==State::Unknown);
  feed(rate,t,8,.1);
  CHECK(rate.state()==State::Ok&&std::abs(rate.hz()-10.)<1e-6);
  rate.add(t+=.4);feed(rate,t,3,.1);                                 // one late scan must not change the verdict
  CHECK(rate.state()==State::Ok);
  ScanRate slow(7.,4.);double u=0.;slow.add(u);feed(slow,u,12,.5);   // the 2 Hz sighting
  CHECK(slow.state()==State::TooSlow&&std::abs(slow.hz()-2.)<1e-6);
  ScanRate mid(7.,4.);double v=0.;mid.add(v);feed(mid,v,12,.2);      // 5 Hz: warn only
  CHECK(mid.state()==State::Slow);
  ScanRate loaded(7.,4.);double w=0.;loaded.add(w);feed(loaded,w,12,1./8.11);  // 8.11 Hz measured under load on 2026-11-04
  CHECK(loaded.state()==State::Ok);
  feed(rate,t,11,.5);                                                // rate drops: the window turns over
  CHECK(rate.state()==State::TooSlow);
  rate.add(t-1.);                                                    // time went backwards: measure again
  CHECK(rate.state()==State::Unknown&&std::isnan(rate.last_interval()));
  bool rejected=false;try{ScanRate bad(3.,4.);}catch(const std::invalid_argument&){rejected=true;}
  CHECK(rejected);
  std::cout<<"Scan rate: 10 Hz ok, late scan ignored, 8.11 Hz ok, 5 Hz warn, 2 Hz too slow, window turnover, time reset: PASS\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

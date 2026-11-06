#include "a1_fusion/core.hpp"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) {
  if(argc!=4) return 2;
  try { std::cout<<std::setprecision(17)<<a1::p5(std::stod(argv[1]),std::stod(argv[2]),std::stod(argv[3]))<<'\n'; }
  catch(const std::exception& e){ std::cerr<<e.what()<<'\n';return 2; }
}

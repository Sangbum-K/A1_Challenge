#include "a1_fusion/core.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <iomanip>
// Historical unquoted numeric p5_10hz.csv format. Replays P5, not perception.
std::vector<std::string> split(const std::string& s){
  std::vector<std::string> out;std::stringstream ss(s);std::string v;
  while(std::getline(ss,v,','))out.push_back(v);
  return out;
}
int main(int argc,char** argv){
  if(argc!=3){std::cerr<<"Usage: replay_p5_csv INPUT.csv OUTPUT.csv\n";return 2;}
  try {
    std::ifstream in(argv[1]);if(!in)throw std::runtime_error("input open failed");
    std::ofstream out(argv[2]);if(!out)throw std::runtime_error("output open failed");
    std::string line;std::getline(in,line);auto names=split(line);std::map<std::string,size_t> cols;
    for(size_t i=0;i<names.size();++i)cols[names[i]]=i;
    out<<"sample,p5_available,p5_cpp_mps2,acc_cmd_gate1_mps2\n"<<std::setprecision(17);
    size_t count=0,valid=0;double max_error=0;
    while(std::getline(in,line)){
      const auto row=split(line);++count;
      auto get=[&](const std::string& k){return std::stod(row.at(cols.at(k)));};
      if(row.at(cols.at("p5_available"))!="True"){out<<count<<",false,nan,nan\n";continue;}
      const double result=a1::p5(std::max(0.0,get("ego_forward_kph")/3.6),get("range_m"),std::max(0.0,get("lead_forward_kph")/3.6));
      max_error=std::max(max_error,std::abs(result-get("p5_raw_mps2")));++valid;
      out<<count<<",true,"<<result<<','<<std::clamp(result,-3.0,0.0)<<'\n';
    }
    std::cout<<"samples="<<count<<" valid="<<valid<<" max_abs_p5_error="<<std::setprecision(17)<<max_error<<'\n';
    return valid && max_error<1e-7 ? 0 : 1;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}

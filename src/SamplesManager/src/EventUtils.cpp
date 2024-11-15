//
// Created by Nadrino on 06/03/2024.
//

#include "EventUtils.h"

#include <sstream>


/// Indices
namespace EventUtils{
  std::string Indices::getSummary() const{
    std::stringstream ss;
    ss << "dataset(" << dataset << ")";
    ss << ", " << "entry(" << entry << ")";
    ss << ", " << "sample(" << sample << ")";
    ss << ", " << "bin(" << bin << ")";
    return ss.str();
  }
}


/// Weights
namespace EventUtils{
  std::string Weights::getSummary() const{
    std::stringstream ss;
    ss << "base(" << base << ")";
    ss << ", " << "current(" << current << ")";
    return ss.str();
  }
}


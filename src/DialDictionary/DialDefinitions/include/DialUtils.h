//
// Created by Nadrino on 25/09/2024.
//

#ifndef GUNDAM_DIALUTILS_H
#define GUNDAM_DIALUTILS_H

#include <cmath>

namespace DialUtils{

  struct Range{
    double min{std::nan("")};
    double max{std::nan("")};

    Range() = default;
    Range(double min_, double max_) : min(min_), max(max_) {}
  };

}

#endif //GUNDAM_DIALUTILS_H

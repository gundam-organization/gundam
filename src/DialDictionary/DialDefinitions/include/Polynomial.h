//
// Created by Adrien Blanchet on 10/04/2023.
//

#ifndef GUNDAM_POLYNOMIAL_H
#define GUNDAM_POLYNOMIAL_H

#include "DialBase.h"

#include "array"


template <std::size_t N> class Polynomial : public DialBase {

public:
  Polynomial() = default;

private:
  std::array<double, N> _coefs_;

};


#endif //GUNDAM_POLYNOMIAL_H

//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "Norm.h"


double Norm::evalResponseImpl(const DialInputBuffer& input_) {
  return input_.getBuffer()[0];
}

//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialNorm.h"



double DialNorm::evalResponseImpl(const DialInputBuffer& input_) {
  return input_.getBuffer()[0];
}

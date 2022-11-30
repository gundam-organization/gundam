//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALNORM_H
#define GUNDAM_DIALNORM_H

#include "DialBase.h"

// for norm dial, no cache is needed


class DialNorm : public DialBase {

public:
  DialNorm() = default;

  double evalResponseImpl(const DialInputBuffer& input_) override;

};


#endif //GUNDAM_DIALNORM_H

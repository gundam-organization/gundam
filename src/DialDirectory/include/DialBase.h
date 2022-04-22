//
// Created by Adrien BLANCHET on 13/04/2022.
//

#ifndef GUNDAM_DIALBASE_H
#define GUNDAM_DIALBASE_H

#include "GenericToolbox.h"

#include "vector"

ENUM_EXPANDER( DialType, -1
               , Unset
               , Norm
               , Spline
               , Graph
               , Nested
)

// DialBase is a virtual class which is in charge of providing the response to be applied on the physics events.
// DialApplier is supposed to be owned by DialCollection

class DialBase {

protected:
  // Not supposed to define a bare Dial. Use the downcast instead
  explicit DialBase(DialType dialType_);

public:
  virtual ~DialBase();

  double evalResponse();

private:
  const DialType _dialType_;
  std::shared_ptr<std::mutex> _evalMutex_{nullptr};



};


#endif //GUNDAM_DIALBASE_H

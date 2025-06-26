//
// Created by Nadrino on 26/06/2025.
//

#ifndef BINNORMALISER_H
#define BINNORMALISER_H

#include "ConfigUtils.h"

class BinNormaliser : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  std::string name{};
  GenericToolbox::Range normParameter{};
  std::string disabledBinDim{};
  std::string parSetNormaliserName{};

};



#endif //BINNORMALISER_H

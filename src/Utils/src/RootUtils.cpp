//
// Created by Nadrino on 27/09/2024.
//

#include "RootUtils.h"

#include "TDirectory.h"
#include "TObject.h"

namespace RootUtils{

  bool ObjectReader::quiet{false};
  bool ObjectReader::throwIfNotFound{false};
  bool ObjectReader::readObject(TDirectory* f_, const std::string& objPath_){
    return readObject<TObject>(f_, objPath_);
  }

}



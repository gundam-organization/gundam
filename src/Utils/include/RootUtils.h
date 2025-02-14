//
// Created by Nadrino on 27/09/2024.
//

#ifndef GUNDAM_ROOTUTILS_H
#define GUNDAM_ROOTUTILS_H

#include "TDirectory.h"

#include "GenericToolbox.String.h"
#include "Logger.h"

#include <functional>
#include <string>


namespace RootUtils{

  class ObjectReader{

  public:
    template<typename T> static bool readObject(TDirectory* f_, const std::vector<std::string>& objPathList_, const std::function<void(T*)>& action_ = [](T*){} );
    template<typename T> static bool readObject(TDirectory* f_, const std::string& objPath_, const std::function<void(T*)>& action_ = [](T*){});
    static bool readObject( TDirectory* f_, const std::string& objPath_);

    static bool quiet;
    static bool throwIfNotFound;

  };

  // template impl
  template<typename T> bool ObjectReader::readObject( TDirectory* f_, const std::vector<std::string>& objPathList_, const std::function<void(T*)>& action_ ){
    using namespace GenericToolbox::ColorCodes;
    T* obj;
    for( auto& objPath : objPathList_ ){
      obj = f_->Get<T>(objPath.c_str());
      if( obj != nullptr ){ break; }
    }
    if( obj == nullptr ){
      LogErrorIf(not ObjectReader::quiet) << redLightText << "Could not find object among names: " << resetColor << GenericToolbox::toString(objPathList_) << std::endl;
      LogThrowIf(ObjectReader::throwIfNotFound, "Object not found.");
      return false;
    }
    action_(obj);
    return true;
  }
  template<typename T> bool ObjectReader::readObject(TDirectory* f_, const std::string& objPath_, const std::function<void(T*)>& action_){
    // auto-forwarding to the general case
    return ObjectReader::readObject(f_, std::vector<std::string>{objPath_}, action_);
  }

}

#endif //GUNDAM_ROOTUTILS_H

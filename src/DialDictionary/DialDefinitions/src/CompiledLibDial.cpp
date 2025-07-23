//
// Created by Nadrino on 03/03/2024.
//

#include "CompiledLibDial.h"
#include "GundamGlobals.h"

#include "Logger.h"

#include <dlfcn.h>


double CompiledLibDial::evalResponse( const DialInputBuffer &input_ ) const{
  // Eval with dynamic function
  std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
  return reinterpret_cast<double(*)(double*)>(_evalFct_)((double*) &input_.getInputBuffer()[0]);
}

bool CompiledLibDial::loadLibrary(const std::string& path_){
  LogInfo << "Loading shared lib: " << path_ << std::endl;
  if( not GenericToolbox::isFile(path_) ){
    LogError << "Could not find lib file: " << path_ << std::endl;
    return false;
  }

  _loadedLibrary_ = dlopen(path_.c_str(), RTLD_LAZY );
  if( _loadedLibrary_ == nullptr ){
    LogError << "Cannot open library: " << dlerror() << std::endl;
    return false;
  }

  _evalFct_ = (dlsym(_loadedLibrary_, "evalVariable"));
  if( _evalFct_ == nullptr ){
    LogError << "Cannot open evalFcn" << std::endl;
    return false;
  }

  return true;
}

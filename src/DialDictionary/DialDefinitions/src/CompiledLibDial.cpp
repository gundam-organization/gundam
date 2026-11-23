//
// Created by Nadrino on 03/03/2024.
//

/// @file CompiledLibDial.cpp
/// @brief Implements CompiledLibDial: loading the user library and calling
///        its evaluation function.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in CompiledLibDial.h.
/// The comments here describe how each method does its job.

#include "CompiledLibDial.h"
#include "GundamGlobals.h"

#include "Logger.h"

#include <dlfcn.h>


/// @par Implementation
/// Locks GundamGlobals::getGlobalMutEx() for the whole call, then casts the
/// stored `void*` to `double(*)(double*)` and calls it with the address of the
/// first buffer value.
double CompiledLibDial::evalResponse( const DialInputBuffer &input_ ) const{
  // Eval with dynamic function
  std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
  return reinterpret_cast<double(*)(double*)>(_evalFct_)((double*) &input_.getInputBuffer()[0]);
}

/// @par Implementation
/// Checks that the file exists, opens it with dlopen() and RTLD_LAZY
/// (function symbols resolved on first call), then looks up `evalVariable`
/// with dlsym(). If the symbol is missing, the library stays open and
/// `_loadedLibrary_` keeps its handle.
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

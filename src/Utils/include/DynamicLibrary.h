#ifndef GUNDAM_DYNAMIC_LIBRARY_H
#define GUNDAM_DYNAMIC_LIBRARY_H

#if defined(_WIN32)
#include "GenericToolbox.Windows.h"
#else
#include <dlfcn.h>
#endif

#include <string>

namespace GundamUtils { namespace DynamicLibrary {
#if defined(_WIN32)
  inline DWORD& lastErrorCode(){
    static thread_local DWORD code = ERROR_SUCCESS;
    return code;
  }
#endif

  inline void* open(const char* path_){
#if defined(_WIN32)
    const auto library = LoadLibraryA(path_);
    lastErrorCode() = library == nullptr ? GetLastError() : ERROR_SUCCESS;
    return library;
#else
    return dlopen(path_, RTLD_LAZY);
#endif
  }

  inline void* symbol(void* library_, const char* name_){
#if defined(_WIN32)
    const auto address = GetProcAddress(static_cast<HMODULE>(library_), name_);
    lastErrorCode() = address == nullptr ? GetLastError() : ERROR_SUCCESS;
    return reinterpret_cast<void*>(address);
#else
    return dlsym(library_, name_);
#endif
  }

  inline std::string error(){
#if defined(_WIN32)
    // Logging between the failed operation and this call may change GetLastError.
    const DWORD code = lastErrorCode();
    lastErrorCode() = ERROR_SUCCESS;
    if( code == ERROR_SUCCESS ){ return {}; }
    char message[2048]{};
    const auto length = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                      nullptr, code, 0, message, sizeof(message), nullptr);
    return "Windows error " + std::to_string(code) + (length == 0 ? "" : ": " + std::string(message, length));
#else
    const char* message = dlerror();
    return message == nullptr ? "" : message;
#endif
  }

  inline void close(void* library_){
    if( library_ == nullptr ){ return; }
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(library_));
#else
    dlclose(library_);
#endif
  }
}}

#endif

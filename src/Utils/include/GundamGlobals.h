//
// Created by Nadrino on 11/02/2021.
//

#ifndef GUNDAM_GUNDAM_GLOBALS_H
#define GUNDAM_GUNDAM_GLOBALS_H

#include <algorithm>
#include <mutex>

class GundamGlobals{

public:
  // setters
  static void setIsDebug(bool enable_){ _parameters_.isDebug = enable_; }
  static void setLightOutputMode(bool enable_){ _parameters_.lightOutputModeEnabled = enable_; }
  static void setNumberOfThreads(int nbCpuThreads_){ _parameters_.nbCpuThreads = nbCpuThreads_; }

  // getters
  static bool isDebug(){ return _parameters_.isDebug; }
  static bool isLightOutputMode(){ return _parameters_.lightOutputModeEnabled; }
  static int getNbCpuThreads(int nbMax_ = 0){ return nbMax_ == 0 ? _parameters_.nbCpuThreads : std::min(nbMax_, _parameters_.nbCpuThreads); }

  // Return a truly global mutex for times when things, really, truly, need to
  // be to be mutually exclusive.  A prime example is when something is
  // accessed in an external library (e.g. loaded via dlopen (or the like)),
  // and there is no guarrantee that it is in any way thread safe.  A global
  // lock must be used because symbols in the external libraries may be
  // accessed from different places (for example, the Kriged, Tabulated, and
  // CompiledLib dials might use the same shared object).  NOTE: This is here
  // for "last resort" cases where we have no other guarrantees.  This is not
  // used for the normal GUNDAM threading.
  static std::mutex& getGlobalMutEx() {
    return _parameters_.globalMutEx;
  }

private:

  struct Parameters{
    // Debug mode is set to enable extra printouts during runtime.
    bool isDebug{false};

    // Light output mode will disable the writing of objects in the output
    // files.  This makes the produced output .root file lighter.
    bool lightOutputModeEnabled{false};

    // how many parallel threads must be running during the
    // compatible routines during runtime
    int nbCpuThreads{1};

    // A global mutex for when it's truly needed.
    std::mutex globalMutEx{};
  };
  static Parameters _parameters_;


};

#endif // GUNDAM_GUNDAM_GLOBALS_H

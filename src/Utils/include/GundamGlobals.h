//
// Created by Nadrino on 11/02/2021.
//

#ifndef GUNDAM_GUNDAM_GLOBALS_H
#define GUNDAM_GUNDAM_GLOBALS_H


class GundamGlobals{

public:
  // setters
  static void setIsDebug(bool enable_){ _parameters_.isDebug = enable_; }
  static void setLightOutputMode(bool enable_){ _parameters_.lightOutputModeEnabled = enable_; }
  static void setNumberOfThreads(int nbCpuThreads_){ _parameters_.nbCpuThreads = nbCpuThreads_; }

  // getters
  static bool isDebug(){ return _parameters_.isDebug; }
  static bool isLightOutputMode(){ return _parameters_.lightOutputModeEnabled; }
  static int getNbCpuThreads(){ return _parameters_.nbCpuThreads; }

private:

  struct Parameters{
    // Debug mode is set to enable extra printouts during runtime.
    bool isDebug{false};

    // Light output mode will disable the writing of objects in the output files.
    // This makes the produced output .root file lighter.
    bool lightOutputModeEnabled{false};

    // how many parallel threads must be running during the
    // compatible routines during runtime
    int nbCpuThreads{1};
  };
  static Parameters _parameters_;


};

#endif // GUNDAM_GUNDAM_GLOBALS_H

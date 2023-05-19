//
// Created by Adrien BLANCHET on 21/11/2022.
//

#ifndef GUNDAM_LINKDEF_H
#define GUNDAM_LINKDEF_H


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ defined_in "@CMAKE_SOURCE_DIR@/submodules/cpp-generic-toolbox/include/GenericToolbox.Root.h";
#pragma link C++ defined_in "@CMAKE_SOURCE_DIR@/submodules/cpp-generic-toolbox/include/GenericToolbox.RawDataArray.h";
#pragma link C++ defined_in "@CMAKE_SOURCE_DIR@/submodules/cpp-generic-toolbox/include/GenericToolbox.CorrelatedVariablesSampler.h";
#pragma link C++ defined_in "@CMAKE_SOURCE_DIR@/submodules/simple-cpp-logger/include/Logger.h";

#endif


#endif //GUNDAM_LINKDEF_H

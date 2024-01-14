//
// Created by Nadrino on 02/11/2022.
//

#ifndef GUNDAM_UTILSLINKDEF_H
#define GUNDAM_UTILSLINKDEF_H

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ class DataBin+;
#pragma link C++ class DataBinSet+;
#pragma link C++ class GundamGreetings+;
#pragma link C++ class JsonBaseClass+;
#pragma link C++ class VariableDictionary+;
#pragma link C++ class YamlUtils+;

#endif

#endif //GUNDAM_UTILSLINKDEF_H

//
// Created by Adrien BLANCHET on 02/11/2022.
//

#ifndef GUNDAM_UTILSLINKDEF_H
#define GUNDAM_UTILSLINKDEF_H

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ class DataBin+;
#pragma link C++ class DataBinSet+;
#pragma link C++ class GundamGreetings+;
#pragma link C++ class JsonBaseClass+;
#pragma link C++ class VariableDictionary+;
#pragma link C++ class YamlUtils+;

//#pragma link C++ defined_in "JsonUtils.h";
//#pragma link C++ defined_in "GenericToolbox.h";
//#pragma link C++ defined_in "GenericToolbox.Root.h";

//#pragma link C++ nestedclass;
//#pragma link C++ nestedtypedef;

//#pragma link C++ namespace GenericToolbox+;
//#pragma link C++ defined_in namespace GenericToolbox+;
//#pragma link C++ namespace JsonUtils+;
//#pragma link C++ defined_in namespace JsonUtils+;

//#pragma link C++ namespace GenericToolbox+;
//#pragma link C++ namespace JsonUtils+;

#endif

#endif //GUNDAM_UTILSLINKDEF_H

//
// Created by Nadrino on 21/02/2025.
//

/// @file DialBuilder.h
/// @brief Declares DialBuilder, an empty placeholder class that nothing uses.
/// @ingroup DialDictionary

#ifndef DIALBUILDER_H
#define DIALBUILDER_H



/// Empty placeholder class: it has no members and no behaviour.
///
/// @note Nothing in GUNDAM uses it. No file includes DialBuilder.h except
///       DialBuilder.cpp, and DialBuilder.cpp is not listed in
///       src/DialDictionary/CMakeLists.txt, so it is not compiled. Dials are
///       built by DialCollection (DialCollection::makeDial()) and, for
///       tabulated and kriged dials, by the dial factories
///       (TabulatedDialFactory, KrigedDialFactory).
/// @ingroup DialDictionary
class DialBuilder {

};



#endif //DIALBUILDER_H

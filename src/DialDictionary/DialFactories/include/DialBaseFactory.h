#ifndef DialBaseFactory_h_Seen
#define DialBaseFactory_h_Seen

#include <DialBase.h>



#include <TObject.h>

#include <string>

// A factory that will build DialBase objects and return the pointer to the
// object.  It's written as a class instead of as a function in case we want
// to close around values, and the normal implemenation would be as a functor,
// but that doesn't work well in some IDEs.  This will use other factories to
// build the dials, so try to keep the interaces uniform (there isn't tight
// inheritance relationship, and C++ isn't good at pure "interface" classes).
// The ownership of the object is passed to the caller so the pointer should
// be put into a managed pointer (e.g. unique_ptr, or shared_ptr).
class DialBaseFactory {
public:
  DialBaseFactory() = default;
  ~DialBaseFactory() = default;

  // Construct a pointer to the correct DialBase.  The dial title is provided
  // so that better error messages can be printed. This uses the dialType and
  // dialSubType to figure out the correct class, and then uses the object
  // pointed to by the dialInitializer to fill the dial.  The initializer is
  // not a constant due to constraints from ROOT.  NOTE: The ownership of the
  // pointer is passed to the caller, so it should be put in a managed
  // variable (e.g. a unique_ptr, or shared_ptr).
  DialBase* makeDial(const std::string& dialTitle_,
                     const std::string& dialType_,
                     const std::string& dialSubType_,
                     TObject* dialInitializer_,
                     bool useCachedDial_);

  DialBase* makeDial(const JsonType& config_);
};

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:

#endif

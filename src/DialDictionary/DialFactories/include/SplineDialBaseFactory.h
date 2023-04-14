#ifndef SplineDialBaseFactory_h_Seen
#define SplineDialBaseFactory_h_Seen

#include <DialBase.h>

#include <TObject.h>

#include <string>


// A factory that handles "dialType: Spline" from the YAML to create a Dial
// and return the pointer to the object.  Try to keep the interaces uniform
// (see DialBaseFactory).  The ownership of the object is passed to the caller
// so the pointer should be put into a managed pointer (e.g. unique_ptr, or
// shared_ptr).
class SplineDialBaseFactory {
public:
  SplineDialBaseFactory() = default;
  ~SplineDialBaseFactory() = default;

  /// Fill the points starting from a TObject that needs to be pointing to a
  /// graph.  This returns false if it can't get the points.
  bool FillFromGraph(std::vector<double>& xPoint,
                     std::vector<double>& yPoint,
                     std::vector<double>& slope,
                     TObject* dialInitializer,
                     const std::string& splType);

  /// Fill the points starting from a TObject that needs to be pointing to a
  /// spline.  This returns false if it can't get the points.
  bool FillFromSpline(std::vector<double>& xPoint,
                      std::vector<double>& yPoint,
                      std::vector<double>& slope,
                      TObject* dialInitializer,
                      const std::string& splType);

  /// Apply the monotonic constraint to the slopes.
  void MakeMonotonic(const std::vector<double>& xPoint,
                     const std::vector<double>& yPoint,
                     std::vector<double>& slope);

  /// Implement the factory that constructs a pointer to the correct
  /// DialBase.  This uses the dialType and dialSubType to figure out the
  /// correct class, and then uses the object pointed to by the
  /// dialInitializer to fill the dial.  The ownership of the pointer is
  /// passed to the caller, so it should be put in a managed variable (e.g. a
  /// unique_ptr, or shared_ptr).
  DialBase* makeDial(const std::string& dialType_,
                     const std::string& dialSubType_,
                     TObject* dialInitializer_,
                     bool useCachedDial_);

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

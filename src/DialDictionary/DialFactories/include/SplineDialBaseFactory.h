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

  // Take vectors of X and Y values and fill anothera vector with the slopes
  // according to the Catmull-Rom prescription.
  void FillCatmullRomSlopes(const std::vector<double>& X,
                            const std::vector<double>& Y,
                            std::vector<double>& slope);

  // Take vectors of X and Y values and fill anothera vector with the slopes
  // according to the Akima prescription.
  void FillAkimaSlopes(const std::vector<double>& X,
                       const std::vector<double>& Y,
                       std::vector<double>& slope);

private:
  std::vector<double> _xPointListBuffer_{};
  std::vector<double> _yPointListBuffer_{};
  std::vector<double> _slopeListBuffer_{};

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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/scripts/gundam-build.sh"
// End:

#endif

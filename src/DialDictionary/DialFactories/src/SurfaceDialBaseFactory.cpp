#include "SurfaceDialBaseFactory.h"

// Explicitly list the headers that are actually needed.  Do not include
// others.
#include "Bilinear.h"
#include "Bicubic.h"

#include <TH2.h>

LoggerInit([]{
  Logger::setUserHeaderStr("[SurfaceFactory]");
});

DialBase* SurfaceDialBaseFactory::makeDial(const std::string& dialTitle_,
                                           const std::string& dialType_,
                                           const std::string& dialSubType_,
                                           TObject* dialInitializer_,
                                           bool useCachedDial_) {

  TH2* srcObject = dynamic_cast<TH2*>(dialInitializer_);

  LogThrowIf(srcObject == nullptr, "Surface dial initializers must be a TH2");

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (dialSubType_ == "Bilinear") {
    // Basic coding: Give a hint to the reader and put likely branch "first".
    // Do we really expect the cached version more than the uncached?
    dialBase = (useCachedDial_) ?
      std::make_unique<BilinearCache>():
      std::make_unique<Bilinear>();
  }
  else if (dialSubType_ == "Bicubic") {
    dialBase = (useCachedDial_) ?
      std::make_unique<BicubicCache>():
      std::make_unique<Bicubic>();
  }

  if (not dialBase) {
    LogError << "Invalid dialSubType value: " << dialSubType_ << std::endl;
    LogError << "Valid dialSubType values are: Bilinear, Bicubic" << std::endl;
    LogThrow("Invalid Surface dialSubType");
  }

  dialBase->buildDial(*srcObject);

  // Pass the ownership without any constraints!
  return dialBase.release();
}

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

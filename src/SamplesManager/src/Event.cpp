//
// Created by Nadrino on 22/07/2021.
//

#include "Event.h"

#include "GundamGlobals.h"
#include "GundamAlmostEqual.h"

#include "Logger.h"

#include <cmath>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Event]"); });
#endif


// misc
std::string Event::getSummary() const {
  std::stringstream ss;
  ss << "Indices{" << _indices_ << "}";
  ss << std::endl << "Weights{" << _weights_ << "}";
  ss << std::endl << "Variables{" << std::endl << _variables_ << std::endl << "}";
  return ss.str();
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
// End:

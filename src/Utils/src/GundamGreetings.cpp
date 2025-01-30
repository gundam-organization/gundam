//
// Created by Adrien BLANCHET on 15/12/2021.
//

#include "GundamGreetings.h"
#include "GundamUtils.h"

#include "GenericToolbox.Os.h"
#include "GenericToolbox.String.h"
#include "Logger.h"

#include <sstream>


void GundamGreetings::hello() {
  std::stringstream ss;
  ss << "Welcome to GUNDAM "
     << (_appName_.empty() ? GenericToolbox::getExecutableName() : _appName_)
     << " v" + GundamUtils::getVersionFullStr();

  LogInfo << GenericToolbox::addUpDownBars(ss.str()) << std::endl;
}
void GundamGreetings::goodbye() {
  std::string goodbyeStr = "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01";
  LogInfo << std::endl << GenericToolbox::repeatString("─", int(goodbyeStr.size())) << std::endl;
  LogInfo << GenericToolbox::makeRainbowString(goodbyeStr, false) << std::endl;
  LogInfo << GenericToolbox::repeatString("─", int(goodbyeStr.size())) << std::endl;
}

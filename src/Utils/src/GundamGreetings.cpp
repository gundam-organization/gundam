//
// Created by Adrien BLANCHET on 15/12/2021.
//

#include "GundamGreetings.h"

#include "versionConfig.h"

#include "Logger.h"
#include "GenericToolbox.h"

#include "sstream"


LoggerInit([]{
  Logger::setUserHeaderStr("[GundamGreetings]");
})

GundamGreetings::GundamGreetings() = default;
GundamGreetings::~GundamGreetings() = default;

void GundamGreetings::setAppName(const std::string &appName) {
  _appName_ = appName;
}

void GundamGreetings::hello() {
  std::stringstream ss;
  ss << "Welcome to the "
  << (_appName_.empty()? GenericToolbox::getExecutableName(): _appName_)
  << " v" + getVersionStr();
  LogInfo << GenericToolbox::repeatString("─", int(ss.str().size())) << std::endl;
  LogInfo << ss.str() << std::endl;
  LogInfo << GenericToolbox::repeatString("─", int(ss.str().size())) << std::endl << std::endl;
}

void GundamGreetings::goodbye() {
  std::string goodbyeStr = "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01";
  LogInfo << std::endl << GenericToolbox::repeatString("─", int(goodbyeStr.size())) << std::endl;
  LogInfo << GenericToolbox::makeRainbowString(goodbyeStr, false) << std::endl;
  LogInfo << GenericToolbox::repeatString("─", int(goodbyeStr.size())) << std::endl;
}

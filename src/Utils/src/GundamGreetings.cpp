//
// Created by Adrien BLANCHET on 15/12/2021.
//

#include "GundamGreetings.h"
#include "GundamUtils.h"

#include "Logger.h"
#include "GenericToolbox.h"

#include <sstream>


LoggerInit([]{
  Logger::setUserHeaderStr("[GundamGreetings]");
});

GundamGreetings::GundamGreetings() = default;
GundamGreetings::~GundamGreetings() = default;

void GundamGreetings::setAppName(const std::string &appName) {
  _appName_ = appName;
}

void GundamGreetings::hello() {
  std::stringstream ss;
  ss << "Welcome to GUNDAM "
     << (_appName_.empty() ? GenericToolbox::getExecutableName() : _appName_)
     << " v" + GundamUtils::getVersionStr();

  LogInfo << GenericToolbox::addUpDownBars(ss.str()) << std::endl;
}
void GundamGreetings::goodbye() {
  std::string goodbyeStr = "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01";
  LogInfo << std::endl << GenericToolbox::repeatString("─", int(goodbyeStr.size())) << std::endl;
  LogInfo << GenericToolbox::makeRainbowString(goodbyeStr, false) << std::endl;
  LogInfo << GenericToolbox::repeatString("─", int(goodbyeStr.size())) << std::endl;
}

bool GundamGreetings::isNewerOrEqualVersion(const std::string &minimalVersion_){
  if( GundamUtils::getVersionStr() == "X.X.X" ){
    LogAlert << "Can't check version requirement. Assuming OK." << std::endl;
    return true;
  }
  auto minVersionSplit = GenericToolbox::splitString(minimalVersion_, ".");
  LogThrowIf(minVersionSplit.size() != 3, "Invalid version format: " << minimalVersion_);
  auto curVersionSplit = GenericToolbox::splitString(GundamUtils::getVersionStr(), ".");
  LogThrowIf(curVersionSplit.size() != 3, "Invalid current version format: " << GundamUtils::getVersionStr());

  // stripping "f" tag
  if( minVersionSplit[2].back() == 'f' ){ minVersionSplit[2].pop_back(); }
  if( curVersionSplit[2].back() == 'f' ){ curVersionSplit[2].pop_back(); }

  if( std::stoi(curVersionSplit[0]) > std::stoi(minVersionSplit[0]) ) return true; // major is GREATER -> OK
  if( std::stoi(curVersionSplit[0]) < std::stoi(minVersionSplit[0]) ) return false; // major is LOWER -> NOT OK
  // major is equal -> next

  if( std::stoi(curVersionSplit[1]) > std::stoi(minVersionSplit[1]) ) return true; // minor is GREATER -> OK
  if( std::stoi(curVersionSplit[1]) < std::stoi(minVersionSplit[1]) ) return false; // minor is LOWER -> NOT OK
  // minor is equal -> next

  if( std::stoi(curVersionSplit[2]) > std::stoi(minVersionSplit[2]) ) return true; // revision is GREATER -> OK
  if( std::stoi(curVersionSplit[2]) < std::stoi(minVersionSplit[2]) ) return false; // revision is LOWER -> NOT OK
  // minor is equal -> OK
  return true;
}
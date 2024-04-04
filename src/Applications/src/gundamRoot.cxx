//
// Created by Adrien BLANCHET on 23/10/2022.
//

#include "GundamUtils.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Utils.h"

#include "TRint.h"

#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>

LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


int main(int argc, char **argv) {

  std::vector<std::string> argvVector(argv, argv + argc);

  argvVector.emplace_back("-n");
  argvVector.emplace_back("-l");

  std::vector<char*> cstrings;
  cstrings.reserve(argvVector.size());
  for(auto& s: argvVector) cstrings.push_back(&s[0]);
  int argcInterpreter = int( cstrings.size() );
  char** argvInterpreter{ cstrings.data() };

  LogInfo << "GUNDAM v" << GundamUtils::getVersionFullStr() << " / ROOT v" << ROOT_RELEASE << std::endl;
  LogInfo << "Creating ROOT interpreter..." << std::endl;
  auto *theApp = new TRint(
      "Rint"
      ,&argcInterpreter
      ,argvInterpreter
      , nullptr/*options*/
      , 0 /*numOptions*/
      , kFALSE /*noLogo*/
//    , kTRUE /*exitOnUnknownArgs*/ // ROOT 6.20?
  );

  LogInfo << "Enabling GenericToolbox lib..." << std::endl;
  gROOT->ProcessLine( Form(".include %s/submodules/cpp-generic-toolbox/include", GundamUtils::getSourceCodePath().c_str()) );
  gROOT->ProcessLine("#include \"GenericToolbox.Utils.h\"");
  gROOT->ProcessLine("#include \"GenericToolbox.Root.h\"");

  LogInfo << "Enabling Logger lib..." << std::endl;
  gROOT->ProcessLine( Form(".include %s/submodules/simple-cpp-logger/include", GundamUtils::getSourceCodePath().c_str()) );
  gROOT->ProcessLine("#include \"Logger.h\"");

  theApp->SetPrompt("gundamRoot [%d] ");

  LogInfo << "Running interpreter..." << std::endl;
  theApp->Run();

  return EXIT_SUCCESS;
}

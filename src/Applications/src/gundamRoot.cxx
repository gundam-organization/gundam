//
// Created by Adrien BLANCHET on 23/10/2022.
//
#include "Logger.h"

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
  theApp->ProcessLine("{ GenericToolbox::Enabler e; }");

  theApp->SetPrompt("gundamRoot [%d] ");

  LogInfo << "Running interpreter..." << std::endl;
  theApp->Run();

  delete theApp;

  return EXIT_SUCCESS;
}

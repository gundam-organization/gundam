//
// Created by Adrien BLANCHET on 23/10/2022.
//
#include "Logger.h"

#include "TRint.h"

#include <cstdlib>
#include "string"
#include "vector"

LoggerInit([]{
  Logger::setUserHeaderStr("[gundamRoot.cxx]");
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
      "Rint",
      &argcInterpreter,
      argvInterpreter,
      /*options*/ nullptr,
      /*numOptions*/ 0,
      /*noLogo*/ kFALSE
//      ,
//      /*exitOnUnknownArgs*/ kTRUE // ROOT 6.20?
  );

//  LogInfo << "Including GenericToolbox.Root.h..." << std::endl;
//  theApp->ProcessLine("#include \"GenericToolbox.Root.h\"");

  LogInfo << "Running interpreter..." << std::endl;
  theApp->Run();

  delete theApp;

  return EXIT_SUCCESS;
}

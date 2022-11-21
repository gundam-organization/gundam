//
// Created by Adrien BLANCHET on 23/10/2022.
//
#include "Logger.h"

#include "TROOT.h"
#include "TRint.h"

#include <cstdlib>


LoggerInit([]{
  Logger::setUserHeaderStr("[gundamRoot.cxx]");
});

TROOT root("Rint","The ROOT Interactive Interface");

int main(int argc, char **argv) {
  LogInfo << "Creating ROOT interpreter..." << std::endl;
  auto* theApp = new TRint("Rint", &argc, argv, nullptr, 0);
  LogInfo << "Including GenericToolbox.Root.h..." << std::endl;
  theApp->ProcessLine("#include \"GenericToolbox.Root.h\"");
  LogInfo << "Running interpreter..." << std::endl;
  theApp->Run();

  delete theApp;
  return EXIT_SUCCESS;
}

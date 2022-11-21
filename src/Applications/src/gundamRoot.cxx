//
// Created by Adrien BLANCHET on 23/10/2022.
//

#include "TROOT.h"
#include "TRint.h"

#include <cstdlib>

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[gundamRoot.cxx]");
});

TROOT root("Rint","The ROOT Interactive Interface");

int main(int argc, char **argv) {
  LogInfo << "Creating ROOT interpreter..." << std::endl;
  auto theApp = TRint("Rint", &argc, argv, nullptr, 0);
  LogInfo << "Including GenericToolbox.Root.h..." << std::endl;
  theApp.ProcessLine("#include \"GenericToolbox.Root.h\"");
  LogInfo << "Running interpreter..." << std::endl;
  theApp.Run();
  return EXIT_SUCCESS;
}

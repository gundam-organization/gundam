//
// Created by Adrien BLANCHET on 23/10/2022.
//

#include "TROOT.h"
#include "TRint.h"

#include <cstdlib>

TROOT root("Rint","The ROOT Interactive Interface");

int main(int argc, char **argv) {
  auto *theApp = new TRint("Rint", &argc, argv, nullptr, 0);
  // Init Intrinsics, build all windows, and enter event loop
  theApp->Run();

  delete theApp;
  return EXIT_SUCCESS;
}

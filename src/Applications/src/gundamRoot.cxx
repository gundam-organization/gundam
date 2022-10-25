//
// Created by Adrien BLANCHET on 23/10/2022.
//

#include <TROOT.h>
#include <TRint.h>

#include "FitterEngine.h"


int main(int argc, char **argv) {

  auto* theApp = new TRint("Stereo ROOT",&argc,argv,nullptr,0);
  theApp->SetPrompt("gundamRoot [%d] ");
//  theApp->ProcessLine("#include<STToolBox.h>");
//  theApp->ProcessLine("STToolBox::set_verbosity_level(1)");
  theApp->Run();
  return 0;

}

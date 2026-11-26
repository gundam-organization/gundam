#include <TDirectory.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <iostream>
#include <string>

void showLogs(){
  auto* file = gDirectory ? gDirectory->GetFile() : nullptr;
  if( file == nullptr ){
    std::cerr << "Select a GUNDAM ROOT file before running showLogs.\n";
    return;
  }

  TTreeReader reader("gundam/runtime/logger", file);
  TTreeReaderValue<std::string> line(reader, "text");
  while( reader.Next() ) std::cout << *line << '\n';
}

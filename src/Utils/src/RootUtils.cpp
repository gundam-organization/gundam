//
// Created by Nadrino on 27/09/2024.
//

#include "RootUtils.h"

#include "GenericToolbox.Fs.h"
#include "GenericToolbox.Root.h"

#include "TCanvas.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TObject.h"

namespace RootUtils{

  void writeCanvas(TDirectory* dir_, TCanvas* canvas_, const std::string& saveName_, bool forceWriteFile_){
    if( dir_ == nullptr or canvas_ == nullptr ){ return; }
    if( auto* file = dir_->GetFile() ){
      canvas_->SetTitle(GenericToolbox::getFileName(file->GetName()).c_str());
    }
    GenericToolbox::writeInTFileWithObjTypeExt(dir_, canvas_, saveName_, forceWriteFile_);
  }

  bool ObjectReader::quiet{false};
  bool ObjectReader::throwIfNotFound{false};
  bool ObjectReader::readObject(TDirectory* f_, const std::string& objPath_){
    return readObject<TObject>(f_, objPath_);
  }

}


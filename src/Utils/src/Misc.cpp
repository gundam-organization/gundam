//
// Created by Adrien Blanchet on 24/01/2023.
//

#include "Misc.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"

#include "TGraph.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[Misc]");
});


namespace Misc{

  bool isGraphValid(TGraph* gr){

    if( gr == nullptr ) return false;
    if( GenericToolbox::isFlatAndOne(gr) ) return false;

    double xBuf{std::nan("")};
    for( int iPt = 0 ; iPt < gr->GetN() ; iPt++ ){
      if( xBuf == gr->GetX()[iPt] ){ return false; }
      xBuf = gr->GetX()[iPt];
    }

    return true;
  }
  bool isSplineValid(TSpline3* sp){
    if( sp == nullptr ) return false;
    if( GenericToolbox::isFlatAndOne(sp) ) return false;

    double xBuf{std::nan("")};
    double x, y;
    for( int iPt = 0 ; iPt < sp->GetNp() ; iPt++ ){
      sp->GetKnot(iPt, x, y);
      if( xBuf == x ){ return false; }
      xBuf = x;
    }

    return true;
  }

}
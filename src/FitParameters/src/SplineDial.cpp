//
// Created by Nadrino on 26/05/2021.
//

#include "FitParameter.h"

#ifndef USE_TSPLINE3_EVAL
#include "CalculateMonotonicSpline.h"
#include "CalculateUniformSpline.h"
#include "CalculateGeneralSpline.h"
#endif

// Unset for this file since the entire file is deprecated.
#ifdef USE_NEW_DIALS
#undef USE_NEW_DIALS
#endif

#include "SplineDial.h"
#include "DialSet.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"

#include "TFile.h"

LoggerInit([](){ Logger::setUserHeaderStr("[SplineDial]"); } );


SplineDial::SplineDial(const DialSet* owner_) : Dial(DialType::Spline, owner_) {}
SplineDial::SplineDial(const DialSet* owner_, const TGraph& graph_): Dial(DialType::Spline, owner_), _spline_(graph_.GetName(), &graph_) {}

void SplineDial::copySpline(const TSpline3* splinePtr_){
  _spline_ = *splinePtr_;
}
void SplineDial::createSpline(TGraph* grPtr_){
  _spline_ = TSpline3(grPtr_->GetName(), grPtr_);
#ifdef ENABLE_SPLINE_DIAL_FAST_EVAL
  fs.stepsize = (_spline_.GetXmax() - _spline_.GetXmin())/((double) grPtr_->GetN());
#endif
}

void SplineDial::initialize() {
  this->Dial::initialize();
  LogThrowIf(_spline_.GetXmin() == _spline_.GetXmax(), "Spline is not valid.");

  // check if prior is out of bounds:
  if(
         this->getEffectiveDialParameter(_owner_->getOwner()->getPriorValue()) < _spline_.GetXmin()
      or this->getEffectiveDialParameter(_owner_->getOwner()->getPriorValue()) > _spline_.GetXmax()
  ){
    LogError << "Prior value of parameter \""
             << _owner_->getOwner()->getTitle()
             << "\" = " << this->getEffectiveDialParameter( _owner_->getOwner()->getPriorValue() )
        << " is out of the spline bounds: " <<  _spline_.GetXmin() << " < X < " << _spline_.GetXmax()
    << std::endl;
    throw std::logic_error("Prior is out of the spline bounds.");
  }

#ifndef USE_TSPLINE3_EVAL
  fillSplineData();
#endif

  try{ this->evalResponse(); }
  catch(...){
    if( not Dial::disableDialCache ) {
      LogError << "Error while evaluating spline response at the prior value: d(" << _dialParameterCache_ << ") = " << _dialResponseCache_ << std::endl;
    }
    throw std::logic_error("error eval spline response");
  }
}

std::string SplineDial::getSummary() {
  std::stringstream ss;
  ss << Dial::getSummary();
  return ss.str();
}
const TSpline3* SplineDial::getSplinePtr() const {
  return &_spline_;
}

double SplineDial::calcDial(double parameterValue_) {
  if( not _owner_->isAllowDialExtrapolation() ){
    if     (parameterValue_ <= _spline_.GetXmin()) { parameterValue_ = _spline_.GetXmin(); }
    else if(parameterValue_ >= _spline_.GetXmax()) { parameterValue_ = _spline_.GetXmax(); }
  }
#ifdef USE_TSPLINE3_EVAL
  return _spline_.Eval(parameterValue_);
#else
  double dialResponse{};
  if (_splineType_ == SplineDial::Uniform) {
      dialResponse = CalculateUniformSpline(
          parameterValue_, -1E20, 1E20,
          _splineData_.data(), int(_splineData_.size()));
  }
  else if (_splineType_ == SplineDial::General) {
      dialResponse = CalculateGeneralSpline(
          parameterValue_, -1E20, 1E20,
          _splineData_.data(), int(_splineData_.size()));
  }
  else if (_splineType_ == SplineDial::Monotonic) {
      dialResponse = CalculateMonotonicSpline(
          parameterValue_, -1E20, 1E20,
          _splineData_.data(), int(_splineData_.size()));
  }
  else if (_splineType_ == SplineDial::ROOTSpline) {
      dialResponse = _spline_.Eval(parameterValue_);
  }
  else {
      LogThrow("Must have a spline type defined");
  }

  // #define SPLINE_DIAL_SLOW_VALIDATION
  #ifdef SPLINE_DIAL_SLOW_VALIDATION
  #error Remove this to compile with validation.
  do {
      double testVal = _spline_.Eval(parameterValue_);
      double avg = std::abs(testVal);
      if (avg < 1.0) avg = 1.0;
      double delta = std::abs(testVal-dialResponse)/avg;
      LogInfo << "VALIDATION: spline"
              << " " << testVal
              << " " << dialResponse
              << " " << delta
              << std::endl;
      if (delta < 1E-6) continue;
      LogInfo << "Bad spline value in SplineDial: " << delta
              << " " << testVal
              << " " << dialResponse
              << std::endl;
      LogThrow("Bad spline value");
  } while (false);
  #endif
  return dialResponse;
#endif
}

void SplineDial::writeSpline(const std::string &fileName_) const{
  TFile* f;
  if(fileName_.empty()) f = TFile::Open(Form("badDial_%p.root", this), "RECREATE");
  else                  f = TFile::Open(fileName_.c_str(), "RECREATE");

  GenericToolbox::writeInTFile( f, &_spline_ );
  f->Close();
}
#ifdef ENABLE_SPLINE_DIAL_FAST_EVAL
void SplineDial::fastEval(){
    //Function takes a spline with equidistant knots and the number of steps
    //between knots to evaluateSpline the spline at some position 'pos'.
    fs.l = int((parameterValue_ - _spline_.GetXmin())
               / fs.stepsize) + 1;

    _spline_.GetCoeff(fs.l, fs.x, fs.y, fs.b, fs.c, fs.d);
    fs.num = parameterValue_ - fs.x;

    if (fs.num < 0){
        fs.l -= 1;
        _spline_.GetCoeff(fs.l, fs.x, fs.y, fs.b, fs.c, fs.d);
        fs.num = parameterValue_ - fs.x;
    }
    _dialResponseCache_ = (fs.y + fs.num * fs.b + fs.num * fs.num * fs.c + fs.num * fs.num * fs.num * fs.d);
}
#endif

#ifndef USE_TSPLINE3_EVAL
const std::vector<double>& SplineDial::getSplineData() const {
  return _splineData_;
}
SplineDial::Subtype SplineDial::getSplineType() const {
  return _splineType_;
}

void SplineDial::fillSplineData() {
    // Check if the spline has uniformly spaced knots.  There is a flag for
    // this is TSpline3, but it's not uniformly (or ever) filled correctly.
    bool uniform = true;
    for (int i = 1; i < _spline_.GetNp()-1; ++i) {
        double x;
        double y;
        _spline_.GetKnot(i-1,x,y);
        double d1 = x;
        _spline_.GetKnot(i,x,y);
        d1 = x - d1;
        double d2 = x;
        _spline_.GetKnot(i+1,x,y);
        d2 = x - d2;
        if (std::abs((d1-d2)/(d1+d2)) > 1E-6) {
            uniform = false;
            break;
        }
    }

    std::string subType = getOwner()->getDialSubType();

    do {
        if (subType == "natural" && fillNaturalSpline(uniform)) break;
        if (subType == "monotonic" && fillMonotonicSpline(uniform)) break;
        fillNaturalSpline(uniform);
    } while(false);

}

bool SplineDial::fillMonotonicSpline(bool uniformKnots) {
    // A monotonic spline has been explicitly requested
    if (!uniformKnots) {
        LogInfo<< "A non uniform monotonic spline dial was requested"
               << ", but when uniform knots are required"
               << std::endl;
        return false;
    }
    _splineType_ = SplineDial::Monotonic;

    // Copy the spline data into local storage.
    _splineData_.push_back(_spline_.GetXmin());
    _splineData_.push_back((_spline_.GetXmax()-_spline_.GetXmin())
                           /(_spline_.GetNp()-1.0));
    for (int i = 0; i < _spline_.GetNp(); ++i) {
        double x;
        double y;
        _spline_.GetKnot(i,x,y);
        _splineData_.push_back(y);
    }
    return true;
}
bool SplineDial::fillNaturalSpline(bool uniformKnots) {
    if (uniformKnots) _splineType_ = SplineDial::Uniform;
    else _splineType_ = SplineDial::General;

    // Copy the spline data into local storage.
    _splineData_.push_back(_spline_.GetXmin());
    _splineData_.push_back((_spline_.GetXmax()-_spline_.GetXmin())
                           /(_spline_.GetNp()-1.0));
    for (int i = 0; i < _spline_.GetNp(); ++i) {
        double x;
        double y;
        _spline_.GetKnot(i,x,y);
        _splineData_.push_back(y);
         _splineData_.push_back(_spline_.Derivative(x));
         if (_splineType_ == SplineDial::Uniform) continue;
        _splineData_.push_back(x);
    }
    return true;
}
#endif

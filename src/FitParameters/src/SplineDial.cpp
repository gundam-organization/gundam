//
// Created by Nadrino on 26/05/2021.
//

#include "TFile.h"

#include "FitParameter.h"
#include "SplineDial.h"

#include "Logger.h"

#include "CalculateMonotonicSpline.h"
#include "CalculateUniformSpline.h"
#include "CalculateGeneralSpline.h"

LoggerInit([](){ Logger::setUserHeaderStr("[SplineDial]"); } )


SplineDial::SplineDial() : Dial(DialType::Spline) {
  this->SplineDial::reset();
}

void SplineDial::reset() {
  this->Dial::reset();
  _spline_ = TSpline3();
}

void SplineDial::initialize() {
  this->Dial::initialize();
  LogThrowIf(_spline_.GetXmin() == _spline_.GetXmax(), "Spline is not valid.");

  // check if prior is out of bounds:
  if(
      _ownerDialSet_->getOwnerFitParameter()->getPriorValue() < _spline_.GetXmin()
      or _ownerDialSet_->getOwnerFitParameter()->getPriorValue() > _spline_.GetXmax()
  ){
    LogError << "Prior value of parameter \""
             << _ownerDialSet_->getOwnerFitParameter()->getTitle()
             << "\" = " << _ownerDialSet_->getOwnerFitParameter()->getPriorValue()
    << " is out of the spline bounds: " <<  _spline_.GetXmin() << " < X < " << _spline_.GetXmax()
    << std::endl;
    throw std::logic_error("Prior is out of the spline bounds.");
  }

  _effectiveDialParameterValue_ = _ownerDialSet_->getOwnerFitParameter()->getPriorValue();

  fillSplineData();

  try{ fillResponseCache(); }
  catch(...){
    LogError << "Error while evaluating spline response at the prior value: d(" << _effectiveDialParameterValue_ << ") = " << _dialResponseCache_ << std::endl;
    throw std::logic_error("error eval spline response");
  }
}

std::string SplineDial::getSummary() {
  std::stringstream ss;
  ss << Dial::getSummary();
  return ss.str();
}

SplineDial::Subtype SplineDial::getSplineType() const {
    return _splineType_;
}

const std::vector<double>& SplineDial::getSplineData() const {
    return _splineData_;
}

void SplineDial::fillResponseCache() {

  if     ( _effectiveDialParameterValue_ < _spline_.GetXmin() ) _dialResponseCache_ = _spline_.Eval(_spline_.GetXmin());
  else if( _effectiveDialParameterValue_ > _spline_.GetXmax() ) _dialResponseCache_ = _spline_.Eval(_spline_.GetXmax());
  else   {
#ifdef USE_INCOMPATIBLE_SPLINES
    _dialResponseCache_ = _spline_.Eval(_effectiveDialParameterValue_);
#else
    if (_splineType_ == SplineDial::Uniform) {
        _dialResponseCache_ = CalculateUniformSpline(
            _effectiveDialParameterValue_, -1E20, 1E20,
            _splineData_.data(), _splineData_.size());
    }
    else if (_splineType_ == SplineDial::General) {
        _dialResponseCache_ = CalculateGeneralSpline(
            _effectiveDialParameterValue_, -1E20, 1E20,
            _splineData_.data(), _splineData_.size());
    }
    else if (_splineType_ == SplineDial::Monotonic) {
        _dialResponseCache_ = CalculateMonotonicSpline(
            _effectiveDialParameterValue_, -1E20, 1E20,
            _splineData_.data(), _splineData_.size());
    }
    else {
        LogThrow("Must have a spline type defined");
    }
#endif
  }

  // #define SPLINE_DIAL_SLOW_VALIDATION
  #ifdef SPLINE_DIAL_SLOW_VALIDATION
  #error Remove this to compile with validation.
  do {
      double testVal = _spline_.Eval(_effectiveDialParameterValue_);
      double avg = std::abs(testVal);
      if (avg < 1.0) avg = 1.0;
      double delta = std::abs(testVal-_dialResponseCache_)/avg;
      LogInfo << "VALIDATION: spline"
              << " " << testVal
              << " " << _dialResponseCache_
              << " " << delta
              << std::endl;
      if (delta < 1E-6) continue;
      LogInfo << "Bad spline value in SplineDial: " << delta
              << " " << testVal
              << " " << _dialResponseCache_
              << std::endl;
      LogThrow("Bad spline value");
  } while (false);
  #endif

  // Checks
  if(_ownerDialSet_->getMinDialResponse() == _ownerDialSet_->getMinDialResponse() and _dialResponseCache_ < _ownerDialSet_->getMinDialResponse() ){
    _dialResponseCache_ = _ownerDialSet_->getMinDialResponse();
  }

  if( _dialResponseCache_ < 0 and _throwIfResponseIsNegative_ ){
    this->writeSpline();
    LogThrow(
      "Negative spline response: dial(" << _effectiveDialParameterValue_ << ") = " << _dialResponseCache_
      << std::endl << "Dial is defined in between: [" << _spline_.GetXmin() << ", " << _spline_.GetXmax() << "]" << std::endl
      << "Parameter: " + _ownerDialSet_->getOwnerFitParameter()->getName() )
  }
}

#ifdef ENABLE_SPLINE_DIAL_FAST_EVAL
void SplineDial::fastEval(){
    //Function takes a spline with equidistant knots and the number of steps
    //between knots to evaluate the spline at some position 'pos'.
    fs.l = int((_effectiveDialParameterValue_ - _spline_.GetXmin())
               / fs.stepsize) + 1;

    _spline_.GetCoeff(fs.l, fs.x, fs.y, fs.b, fs.c, fs.d);
    fs.num = _effectiveDialParameterValue_ - fs.x;

    if (fs.num < 0){
        fs.l -= 1;
        _spline_.GetCoeff(fs.l, fs.x, fs.y, fs.b, fs.c, fs.d);
        fs.num = _effectiveDialParameterValue_ - fs.x;
    }
    _dialResponseCache_ = (fs.y + fs.num * fs.b + fs.num * fs.num * fs.c + fs.num * fs.num * fs.num * fs.d);
}
#endif

void SplineDial::copySpline(const TSpline3* splinePtr_){
  LogThrowIf(_spline_.GetXmin() != _spline_.GetXmax(), "Spline already set")
  _spline_ = *splinePtr_;
}

void SplineDial::createSpline(TGraph* grPtr_){
  LogThrowIf(_spline_.GetXmin() != _spline_.GetXmax(), "Spline already set")
  _spline_ = TSpline3(grPtr_->GetName(), grPtr_);
//  fs.stepsize = (_spline_.GetXmax() - _spline_.GetXmin())/((double) grPtr_->GetN());
}

const TSpline3* SplineDial::getSplinePtr() const {
  return &_spline_;
}

void SplineDial::writeSpline(const std::string &fileName_) const{
  TFile* f;
  if(fileName_.empty()) f = TFile::Open(Form("badDial_%p.root", this), "RECREATE");
  else                  f = TFile::Open(fileName_.c_str(), "RECREATE");

  f->WriteObject(&_spline_, _spline_.GetName());
  f->Close();
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

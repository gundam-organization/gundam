//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_SPLINEDIAL_H
#define GUNDAM_SPLINEDIAL_H

#include "Dial.h"

#include "TSpline.h"

#include "memory"
#include "string"

#define USE_TSPLINE3_EVAL

class SplineDial : public Dial {

public:
  SplineDial();
  std::unique_ptr<Dial> clone() const override { return std::make_unique<SplineDial>(*this); }

  void reset() override;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  void initialize() override;

  const TSpline3* getSplinePtr() const;
  std::string getSummary() override;

  double calcDial(double parameterValue_) override;

  // Debug
  void writeSpline(const std::string &fileName_) const override;

#ifdef ENABLE_SPLINE_DIAL_FAST_EVAL
  void fastEval();
#endif


protected:
  // The representation of the spline read from a root input file.
  TSpline3 _spline_;

#ifdef ENABLE_SPLINE_DIAL_FAST_EVAL
  struct FastSpliner{
    double x, y, b, c, d, num;
    double stepsize{-1};
    int l;
  };
  FastSpliner fs;
#endif

#ifndef USE_TSPLINE3_EVAL
public:
  typedef enum {
    Undefined,  // Fall back to using plain old TSpline3
    Monotonic,  // Use a monotonic cubic spline (only uses function value)
    Uniform,    // Use a cubic spline with uniformly spaced knots
    General,    // A general cubic spline
  } Subtype;

  Subtype getSplineType() const;
  const std::vector<double>& getSplineData() const;

protected:
  // The type of spline that should be used for this dial.
  Subtype _splineType_;

  // A block of data to calculate the spline values.  This can be copied to
  // the Cache::Manager and lets the same spline calculation be used here and
  // there.
  std::vector<double> _splineData_;

  // This fills _splineData_ and sets the _splineType_.  It uses the spline
  // knot spacing, and the spline subtype set in the DialSet (which is read
  // from the configuration file) to determine the spline type.
  void fillSplineData();

  // Fill the spline data for a monotonic spline using the knots in _spline_.
  // This sets _splineType_ to be Monotonic.  If uniformKnots is false, then
  // this will return false (i.e. an error).
  bool fillMonotonicSpline(bool uniformKnots);

  // Fill the spline data for a natural spline using the knots in _spline_.
  // This will generate either a Uniform spline if uniformKnots is true, or a
  // General if it is false.  The knots are not checked to make sure they are
  // actually uniform.
  bool fillNaturalSpline(bool uniformKnots);
#endif

  // DEBUG

};
#endif //GUNDAM_SPLINEDIAL_H

//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_SPLINEDIAL_H
#define GUNDAM_SPLINEDIAL_H

#include "memory"

#include "TSpline.h"

#include "Dial.h"

class SplineDial : public Dial {

public:
  SplineDial();

  void reset() override;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  void initialize() override;

  std::string getSummary() override;
  const TSpline3* getSplinePtr() const;

  // Debug
  void writeSpline(const std::string &fileName_ = "") const;

  void fastEval();

  typedef enum {
       Undefined,  // Fall back to using plain old TSpline3
       Monotonic,  // Use a monotonic cubic spline (only uses function value)
       Uniform,    // Use a cubic spline with uniformly spaced knots
       General,    // A general cubic spline
  } Subtype;

  Subtype getSplineType() const;

  const std::vector<double>& getSplineData() const;

protected:
  void fillResponseCache() override;

private:
  bool _throwIfResponseIsNegative_{true};

  // The representation of the spline read from a root input file.
  TSpline3 _spline_;

  // The type of spline that should be used for this dial.
  Subtype _splineType_;

  struct FastSpliner{
    double x, y, b, c, d, num;
    double stepsize{-1};
    int l;
  };
  FastSpliner fs;

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

  // DEBUG

};
#endif //GUNDAM_SPLINEDIAL_H

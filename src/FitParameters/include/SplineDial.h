//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_SPLINEDIAL_H
#define GUNDAM_SPLINEDIAL_H

#include "Dial.h"

#include "TSpline.h"

#include <memory>
#include <string>

#ifdef USE_NEW_DIALS
#warning Not used with new dial implementation
#endif

class SplineDial : public Dial {

public:
  explicit SplineDial(const DialSet* owner_);
  explicit SplineDial(const DialSet* owner_, const TGraph& graph_);
  [[nodiscard]] std::unique_ptr<Dial> clone() const override { return std::make_unique<SplineDial>(*this); }

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  void initialize() override;

  [[nodiscard]] const TSpline3* getSplinePtr() const;
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

#ifndef USE_TSPLINE3_EVAL // aka with CacheManager
public:
  typedef enum {
    Undefined,  // This should not occur
    ROOTSpline, // Use ROOTs cubic spline (i.e. TSpline3), cannot use with GPU
    Monotonic,  // Use a monotonic cubic spline (only uses function value)
    Uniform,    // Use a cubic spline with uniformly spaced knots
    General,    // A general cubic spline
  } Subtype;

  Subtype getSplineType() const;
  const std::vector<double>& getSplineData() const;

protected:
  // The type of spline that should be used for this dial.
  Subtype _splineType_{ROOTSpline};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
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

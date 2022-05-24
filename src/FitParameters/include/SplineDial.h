//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_SPLINEDIAL_H
#define GUNDAM_SPLINEDIAL_H

#include "Dial.h"

#include "TSpline.h"

#include "memory"
#include "string"


class SplineDial : public Dial {

public:
  SplineDial();
  std::unique_ptr<Dial> clone() const override { return std::make_unique<SplineDial>(*this); }

  void reset() override;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  void initialize() override;

  std::string getSummary() override;
  const TSpline3* getSplinePtr() const;

  // Debug
  void writeSpline(const std::string &fileName_ = "") const;

//  void fastEval();

protected:
  void fillResponseCache() override;


private:
  bool _throwIfResponseIsNegative_{true};
  TSpline3 _spline_;

//  struct FastSpliner{
//    double x, y, b, c, d, num;
//    double stepsize{-1};
//    int l;
//  };
//  FastSpliner fs;

  // DEBUG

};
#endif //GUNDAM_SPLINEDIAL_H

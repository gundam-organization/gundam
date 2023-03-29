//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_SPLINE_H
#define GUNDAM_SPLINE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "TSpline.h"

class Spline : public DialBase {

public:
  Spline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Spline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Spline"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateSpline(input_); }

  void setAllowExtrapolation(bool allowExtrapolation);
  void setSpline(const TSpline3 &spline);
  [[nodiscard]] const TSpline3 &getSpline() const;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  [[nodiscard]] double evaluateSpline(const DialInputBuffer& input_) const;

protected:
  bool _allowExtrapolation_{false};
  TSpline3 _spline_{};

};


#endif //GUNDAM_SPLINE_H

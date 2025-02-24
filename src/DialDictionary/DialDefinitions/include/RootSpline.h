//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_ROOT_SPLINE_H
#define GUNDAM_ROOT_SPLINE_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include "SplineUtils.h"

#include "TSpline.h"

class RootSpline : public DialBase {

public:
  RootSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootSpline"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;

  /// Pass information to the dial so that it can build it's
  /// internal information.  New build overloads should be
  /// added as we have classes of dials
  /// (e.g. multi-dimensional dials).
  virtual void buildDial(const TGraph& grf, const std::string& option_="") override;
  virtual void buildDial(const TSpline3& spl, const std::string& option_="") override;

  void buildDial(const std::vector<SplineUtils::SplinePoint>& splinePointList_);

protected:
  void setSpline(const TSpline3 &spline);
  [[nodiscard]] const TSpline3 &getSpline() const;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  bool _allowExtrapolation_{false};
  TSpline3 _spline_{};

};

#endif // GUNDAM_ROOT_SPLINE_H

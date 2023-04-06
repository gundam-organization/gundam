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
  double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;

  /// Pass information to the dial so that it can build it's
  /// internal information.  New build overloads should be
  /// added as we have classes of dials
  /// (e.g. multi-dimensional dials).
  virtual void buildDial(const TGraph& grf, const std::string &option="") override;
  virtual void buildDial(const TSpline3& spl, const std::string &option_="") override;
  virtual void buildDial(const std::vector<double>& v1,
                         const std::vector<double>& v2,
                         const std::vector<double>& v3,
                         const std::string &option="") override;

protected:
  void setSpline(const TSpline3 &spline);
  [[nodiscard]] const TSpline3 &getSpline() const;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  bool _allowExtrapolation_{false};
  TSpline3 _spline_{};

};


#endif //GUNDAM_SPLINE_H

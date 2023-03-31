//
// Created by Adrien Blanchet on 19/01/2023.
//

#ifndef GUNDAM_LIGHTGRAPH_H
#define GUNDAM_LIGHTGRAPH_H

#include "DialBase.h"
#include "TGraph.h"
#include "vector"


class LightGraph : public DialBase {

public:
  LightGraph() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<LightGraph>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"LightGraph"}; }
  double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;
  bool getAllowExtrapolation() const override;

  virtual void buildDial(const TGraph& grf, std::string option="") override;

protected:
  [[nodiscard]] double evaluateGraph(const DialInputBuffer& input_) const;
  bool _allowExtrapolation_{false};
  Int_t nPoints{0};   ///< Number of points <= fMaxSize
  std::vector<Double_t> xPoints; ///<[fNpoints] array of X points
  std::vector<Double_t> yPoints; ///<[fNpoints] array of Y points
};

#endif //GUNDAM_LIGHTGRAPH_H

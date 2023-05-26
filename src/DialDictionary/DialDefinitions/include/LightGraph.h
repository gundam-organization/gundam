//
// Created by Adrien Blanchet on 19/01/2023.
//

#ifndef GUNDAM_LIGHTGRAPH_H
#define GUNDAM_LIGHTGRAPH_H

#include "DialBase.h"

#include "TGraph.h"

#include <vector>


/// A DialBase class to do piecewise linear interpolation.  This is
/// initialized using a ROOT TGraph, and will return the same values, but uses
/// much less space and is generally faster.
class LightGraph : public DialBase {

public:
  LightGraph() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<LightGraph>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"LightGraph"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;
  [[nodiscard]] bool getAllowExtrapolation() const override;

  virtual void buildDial(const TGraph& grf, const std::string& option_="") override;

  const std::vector<double>& getDialData() const override {return _Data_;}

protected:
  bool _allowExtrapolation_{false};

  // The data for the graph packed as {y0,x0,y1,x1,y2,x2,...}
  std::vector<Double_t> _Data_;
};

typedef CachedDial<LightGraph> LightGraphCache;
#endif //GUNDAM_LIGHTGRAPH_H

//
// Created by Adrien Blanchet on 19/01/2023.
//

#ifndef GUNDAM_ROOT_GRAPH_H
#define GUNDAM_ROOT_GRAPH_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include "TGraph.h"


class RootGraph : public DialBase {

public:
  RootGraph() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootGraph>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"TGraph"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;
  [[nodiscard]] bool getAllowExtrapolation() const override;

  virtual void buildDial(const TGraph& grf, const std::string& option_="") override;

protected:
  [[nodiscard]] double evaluateGraph(const DialInputBuffer& input_) const;

  bool _allowExtrapolation_{false};
  TGraph _graph_{};
};

#endif //GUNDAM_ROOT_GRAPH_H

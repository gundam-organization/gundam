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

  // mandatory overrides
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootGraph>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootGraph"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  // other overrides
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  void setGraph(const TGraph &graph){ _graph_ = graph; }

private:
  bool _allowExtrapolation_{false};
  TGraph _graph_{};
};

#endif //GUNDAM_ROOT_GRAPH_H

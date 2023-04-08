//
// Created by Adrien Blanchet on 07/12/2022.
//

#ifndef GUNDAM_GRAPH_H
#define GUNDAM_GRAPH_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "TGraph.h"

class Graph : public DialBase {

public:
  Graph() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Graph>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Graph"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;
  [[nodiscard]] bool getAllowExtrapolation() const override;

  virtual void buildDial(const TGraph& grf, const std::string& option_="") override;

protected:
  [[nodiscard]] double evaluateGraph(const DialInputBuffer& input_) const;

  bool _allowExtrapolation_{false};
  TGraph _graph_{};
};

typedef CachedDial<Graph> GraphCache;


#endif //GUNDAM_GRAPH_H

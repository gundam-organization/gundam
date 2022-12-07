//
// Created by Adrien Blanchet on 07/12/2022.
//

#ifndef GUNDAM_GRAPH_H
#define GUNDAM_GRAPH_H

#include "DialBase.h"
#include "GraphHandler.h"


class Graph : public DialBase, public GraphHandler {

public:
  Graph() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Graph>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Graph"}; }
  double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateGraph(input_); }

};


#endif //GUNDAM_GRAPH_H

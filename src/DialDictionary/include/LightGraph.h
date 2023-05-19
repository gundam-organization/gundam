//
// Created by Adrien Blanchet on 19/01/2023.
//

#ifndef GUNDAM_LIGHTGRAPH_H
#define GUNDAM_LIGHTGRAPH_H

#include "DialBase.h"
#include "LightGraphHandler.h"

class LightGraph : public DialBase, public LightGraphHandler {

  public:
  LightGraph() = default;

    [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<LightGraph>(*this); }
    [[nodiscard]] std::string getDialTypeName() const override { return {"LightGraph"}; }
    double evalResponseImpl(const DialInputBuffer& input_) override { return this->evaluateGraph(input_); }

};


#endif //GUNDAM_LIGHTGRAPH_H

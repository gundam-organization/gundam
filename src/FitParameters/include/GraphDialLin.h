//
// Created by Adrien BLANCHET on 02/12/2021.
//

#ifndef GUNDAM_GRAPHDIALLIN_H
#define GUNDAM_GRAPHDIALLIN_H

#include "TGraph.h"

#include "Dial.h"

class GraphDialLin : public Dial {

public:
  GraphDialLin();

  void reset() override;
  std::unique_ptr<Dial> clone() const override { return std::make_unique<GraphDialLin>(*this); }

  void setGraph(const TGraph &graph);

  void initialize() override;
//  double evalResponse(const double &parameterValue_) override;
  void fillResponseCache() override;
  std::string getSummary() override;

private:
  TGraph _graph_;
};


#endif //GUNDAM_GRAPHDIALLIN_H

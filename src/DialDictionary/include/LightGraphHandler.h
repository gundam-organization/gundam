//
// Created by Adrien Blanchet on 19/01/2023.
//

#ifndef GUNDAM_LIGHTGRAPHHANDLER_H
#define GUNDAM_LIGHTGRAPHHANDLER_H

#include "DialInputBuffer.h"

#include "TGraph.h"

#include "vector"

class LightGraphHandler {

public:
  LightGraphHandler() = default;
  virtual ~LightGraphHandler() = default;

  void setAllowExtrapolation(bool allowExtrapolation);
  void setGraph(TGraph &graph);

  [[nodiscard]] double evaluateGraph(const DialInputBuffer& input_) const;

protected:
  bool _allowExtrapolation_{false};
  Int_t nPoints;   ///< Number of points <= fMaxSize
  std::vector<Double_t> xPoints; ///<[fNpoints] array of X points
  std::vector<Double_t> yPoints; ///<[fNpoints] array of Y points


};


#endif //GUNDAM_LIGHTGRAPHHANDLER_H

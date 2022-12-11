//
// Created by Adrien Blanchet on 07/12/2022.
//

#ifndef GUNDAM_GRAPHHANDLER_H
#define GUNDAM_GRAPHHANDLER_H

#include "DialInputBuffer.h"

#include "TGraph.h"


class GraphHandler {


public:
  GraphHandler() = default;
  virtual ~GraphHandler() = default;

  void setAllowExtrapolation(bool allowExtrapolation);
  void setGraph(const TGraph &graph);

  [[nodiscard]] double evaluateGraph(const DialInputBuffer& input_) const;

protected:
  bool _allowExtrapolation_{false};
  TGraph _graph_{};


};


#endif //GUNDAM_GRAPHHANDLER_H

#include "BackendModel.h"

void Backends::BackendModel::clear() {
  events.clear();
  eventDials.clear();
  samples.clear();
  parameters.clear();
  totalBins = 0;
}

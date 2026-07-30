#include "BackendEngineView.h"

void Backends::BackendPropagationView::clear() {
  events.clear();
  eventDials.clear();
  inputBuffers.clear();
  samples.clear();
  parameters.clear();
  totalBins = 0;
}

void Backends::BackendEngineView::clear() {
  propagation.clear();
  likelihood.samples.clear();
}

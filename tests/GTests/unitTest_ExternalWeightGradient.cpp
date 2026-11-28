#include "ExternalWeightDialFactory.h"
#include "ExternalWeightDispatcher.h"
#include "Event.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
  class GradientWorker : public ExternalWeightWorker {
  public:
    GradientWorker(){ setSupportsGradient(true); }
    bool partial{false};
    bool invalidWeight{false};
    bool hasGradientStorage() const { return getGradientBuffer() != nullptr; }
    void disableGradient(){ setSupportsGradient(false); }
  private:
    void evaluateImpl(const DialInputBuffer& input_) override {
      const auto& values = input_.getInputBuffer();
      for( size_t row = 0; row < getWeightCount(); ++row ){
        getWeightBuffer()->ptr[row] = values[0] * values[0] + (row + 1) * values[1];
        if( isGradientRequested() ){
          getGradientBuffer()->ptr[2 * row] = 2 * values[0];
          if( not partial ){ getGradientBuffer()->ptr[2 * row + 1] = row + 1; }
        }
      }
      if( invalidWeight ){ getWeightBuffer()->ptr[0] = std::numeric_limits<double>::infinity(); }
    }
  };
}

TEST(ExternalWeightGradient, EventAndBinnedStorageLazyRequestsFailuresAndLifetime) {
  for( bool binned : {false, true} ){
    auto worker = std::make_unique<GradientWorker>();
    ConfigReader config(JsonType{{"useBinnedWeights", binned}, {"inputEventVarList", JsonType::array()},
        {"binning", {{"binningDefinition", {{{"name", "x"}, {"edges", {0., 1., 2.}}}}}}}});
    config.defineFields({{"useBinnedWeights"}, {"inputEventVarList"}, {"binning"}});
    worker->configure(config);
    worker->initialize();
    Event event;
    if( not binned ){ worker->registerEvent(event); worker->registerEvent(event); }
    auto source = worker->getWeightSource();
    ExternalWeightDispatcher first(source, 0), second(source, 1);
    std::vector<ParameterSet> parameters(1);
    for( int i = 0; i < 2; ++i ){ parameters[0].getParameterList().emplace_back(&parameters[0]); }
    parameters[0].getParameterList()[0].setParameterValue(.2);
    parameters[0].getParameterList()[1].setParameterValue(.3);
    DialInputBuffer input;
    input.setParSetRef(&parameters);
    for( int i = 0; i < 2; ++i ){
      DialInputBuffer::ParameterReference ref;
      ref.parSetIndex = 0; ref.parIndex = i;
      input.addParameterReference(ref);
    }
    input.initialise();
    input.update();
    worker->updateWeights(input);
    EXPECT_FALSE(worker->hasGradientStorage());
    EXPECT_FALSE(source->hasCurrentGradient());
    input.update(); // Request at unchanged parameters must bypass the cache.
    const auto revision = input.getRevision();
    ASSERT_TRUE(first.hasAnalyticGradient());
    EXPECT_THROW(static_cast<void>(first.evalGradients(input)), std::runtime_error);
    worker->updateWeights(input);
    EXPECT_GT(input.getRevision(), revision);
    ASSERT_TRUE(source->hasCurrentGradient());
    EXPECT_EQ((std::vector<double>{.4, 1.}), first.evalGradients(input));
    EXPECT_EQ((std::vector<double>{.4, 2.}), second.evalGradients(input));
    EXPECT_EQ(.4, first.evalGradient(input, 0));
    auto generation = source->getGeneration();
    input.update();
    worker->updateWeights(input);
    EXPECT_EQ(generation, source->getGeneration());
    DialInputBuffer wrong;
    wrong.setInputValues({.3, .3});
    EXPECT_THROW(static_cast<void>(first.evalGradients(wrong)), std::runtime_error);
    ExternalWeightDispatcher invalid(source, 2);
    EXPECT_THROW(static_cast<void>(invalid.evalGradients(input)), std::runtime_error);

    // A subsequent value-only publication must invalidate the old matrix.
    parameters[0].getParameterList()[0].setParameterValue(.35);
    input.update();
    worker->updateWeights(input);
    EXPECT_FALSE(source->hasCurrentGradient());
    EXPECT_THROW(static_cast<void>(first.evalGradients(input)), std::runtime_error);
    ASSERT_TRUE(first.hasAnalyticGradient());
    input.update();
    worker->updateWeights(input);
    EXPECT_EQ((std::vector<double>{.7, 1.}), first.evalGradients(input));
    generation = source->getGeneration();

    parameters[0].getParameterList()[0].setParameterValue(.5);
    input.update();
    worker->partial = true;
    ASSERT_TRUE(first.hasAnalyticGradient());
    EXPECT_THROW(worker->updateWeights(input), std::runtime_error);
    EXPECT_FALSE(source->hasCurrentGradient());
    EXPECT_EQ(generation, source->getGeneration());
    EXPECT_THROW(static_cast<void>(first.evalGradients(input)), std::runtime_error);
    worker->partial = false;
    input.update(); // Retry even though the parameter update is now consumed.
    worker->updateWeights(input);
    EXPECT_EQ((std::vector<double>{1., 2.}), second.evalGradients(input));
    worker->invalidWeight = true;
    parameters[0].getParameterList()[0].setParameterValue(.6);
    input.update();
    ASSERT_TRUE(first.hasAnalyticGradient());
    EXPECT_THROW(worker->updateWeights(input), std::runtime_error);
    EXPECT_FALSE(source->hasCurrentGradient());
    worker->invalidWeight = false;
    worker->updateWeights(input);
    worker.reset();
    // The dispatcher owns both shared mappings independently of the worker.
    EXPECT_EQ((std::vector<double>{1.2, 1.}), first.evalGradients(input));
  }
}

TEST(ExternalWeightGradient, UnsupportedWorkerDoesNotAdvertiseGradient) {
  GradientWorker worker;
  worker.disableGradient();
  ExternalWeightDispatcher dial(worker.getWeightSource(), 0);
  EXPECT_FALSE(dial.hasAnalyticGradient());
  DialInputBuffer input;
  EXPECT_THROW(static_cast<void>(dial.evalGradients(input)), std::runtime_error);
}

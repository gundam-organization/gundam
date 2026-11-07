#include "TabulatedDialFactory.h"
#include "KrigedDialFactory.h"
#include "Tabulated.h"
#include "Event.h"
#include "Kriged.h"

#include <gtest/gtest.h>

#include <future>
#include <limits>

namespace {
  int fillValues(const char*, double* table_, int bins_, const double* par_, int){
    for( int i = 0; i < bins_; ++i ){
      table_[i] = (i + 1.) * par_[0] * par_[0] + (i - 1.) * par_[1];
    }
    return 0;
  }
  int fillDerivatives(const char*, double* derivatives_, int bins_, const double* par_, int npar_){
    for( int i = 0; i < bins_; ++i ){
      derivatives_[i * npar_] = 2. * (i + 1.) * par_[0];
      derivatives_[i * npar_ + 1] = i - 1.;
    }
    return 0;
  }
}

TEST(TableGradient, InterpolationMatchesFiniteDifferencesAndClonesShareUpdates){
  std::vector<double> table(4);
  auto source = std::make_shared<TableGradientBuffer>(fillDerivatives);
  Tabulated tabulated(&table, 1, .25, "", source);
  Kriged kriged(&table, 3, {0, 2, 3}, {.5, -.25, .75}, "", source);
  auto clone = kriged.clone();
  for( const auto& point : {std::vector<double>{.3, -.2}, {0., .4}, {-1.2, 0.}} ){
    DialInputBuffer input;
    input.setInputValues(point);
    ASSERT_TRUE(tabulated.hasAnalyticGradient());
    ASSERT_TRUE(kriged.hasAnalyticGradient());
    source->update(table, point, fillValues, "test");
    for( const DialBase* dial : {static_cast<const DialBase*>(&tabulated), static_cast<const DialBase*>(&kriged), static_cast<const DialBase*>(clone.get())} ){
      const auto gradient = dial->evalGradients(input);
      ASSERT_EQ(2U, gradient.size());
      for( int p = 0; p < 2; ++p ){
        auto up = point, down = point;
        up[p] += 1E-6;
        down[p] -= 1E-6;
        fillValues("test", table.data(), int(table.size()), up.data(), 2);
        const double upper = dial->evalResponse(input);
        fillValues("test", table.data(), int(table.size()), down.data(), 2);
        const double lower = dial->evalResponse(input);
        EXPECT_NEAR(gradient[p], (upper - lower) / 2E-6, 1E-8);
        EXPECT_DOUBLE_EQ(gradient[p], dial->evalGradient(input, p));
      }
      fillValues("test", table.data(), int(table.size()), point.data(), 2);
    }
    // Published derivatives can be read concurrently by event workers.
    auto job = std::async(std::launch::async, [&]{ return clone->evalGradients(input); });
    EXPECT_EQ(kriged.evalGradients(input), job.get());
    EXPECT_THROW(static_cast<void>(tabulated.evalGradient(input, -1)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(kriged.evalGradient(input, 2)), std::runtime_error);
  }
}

TEST(TableGradient, RejectsMissingStalePartialAndFailedDerivatives){
  std::vector<double> table(4);
  Tabulated unsupported(&table, 0, .5);
  Kriged unsupportedKriged(&table, 1, {0}, {1.});
  EXPECT_FALSE(unsupported.hasAnalyticGradient());
  EXPECT_FALSE(unsupportedKriged.hasAnalyticGradient());
  DialInputBuffer input;
  input.setInputValues({.3, -.2});
  EXPECT_THROW(static_cast<void>(unsupported.evalGradients(input)), std::runtime_error);
  auto source = std::make_shared<TableGradientBuffer>(fillDerivatives);
  Tabulated dial(&table, 0, .5, "", source);
  EXPECT_THROW(static_cast<void>(dial.evalGradients(input)), std::runtime_error);
  ASSERT_TRUE(dial.hasAnalyticGradient());
  source->update(table, input.getInputBuffer(), fillValues, "test");
  input.setInputValues({.4, -.2});
  EXPECT_THROW(static_cast<void>(dial.evalGradients(input)), std::runtime_error);
  source->update(table, input.getInputBuffer(), fillValues, "test");
  EXPECT_THROW(static_cast<void>(dial.evalGradients(input)), std::runtime_error); // value-only update

  const TableGradientBuffer::UpdateFunction partial = [](const char*, double* values_, int, const double*, int){
    values_[0] = 1.; return 0;
  };
  const TableGradientBuffer::UpdateFunction failed = [](const char*, double*, int, const double*, int){ return 1; };
  const TableGradientBuffer::UpdateFunction nonfinite = [](const char*, double* values_, int bins_, const double*, int npar_){
    std::fill(values_, values_ + bins_ * npar_, std::numeric_limits<double>::infinity()); return 0;
  };
  for( auto function : {partial, failed, nonfinite} ){
    auto invalid = std::make_shared<TableGradientBuffer>(function);
    ASSERT_TRUE(invalid->requestGradient());
    EXPECT_THROW(invalid->update(table, input.getInputBuffer(), fillValues, "broken"), std::runtime_error);
    EXPECT_THROW(invalid->validate(input.getInputBuffer(), table.size()), std::runtime_error);
  }
  ASSERT_TRUE(source->requestGradient());
  EXPECT_THROW(source->update(table, input.getInputBuffer(), partial, "broken values"), std::runtime_error);
  EXPECT_THROW(static_cast<void>(dial.evalGradients(input)), std::runtime_error);
  // Failed updates keep the request pending, so retrying at the same point works.
  source->update(table, input.getInputBuffer(), fillValues, "test");
  EXPECT_NO_THROW(static_cast<void>(dial.evalGradients(input)));
  ASSERT_TRUE(source->requestGradient());
  EXPECT_THROW(source->update(table, input.getInputBuffer(), failed, "broken update"), std::runtime_error);
  EXPECT_THROW(static_cast<void>(dial.evalGradients(input)), std::runtime_error);
}

TEST(TableGradient, FactoriesLoadOptionalGradientSymbolAndPublishCurrentPoint){
  const auto makeConfig = [](const JsonType& table_){
    ConfigReader config(JsonType{{"tableConfig", table_}});
    // Factories normally receive a reader whose parent fields were declared
    // by DialCollection. Reproduce that contract when constructing directly.
    config.defineFields({{FieldFlag::MANDATORY, "tableConfig"}});
    config.checkConfiguration();
    return config;
  };
  JsonType table{{"name", "fixture"}, {"libraryPath", GUNDAM_TABLE_GRADIENT_PLUGIN},
      {"initFunction", "initializeGradientTable"}, {"initArguments", JsonType::array()},
      {"updateFunction", "updateGradientTable"}, {"gradientFunction", "differentiateGradientTable"},
      {"binningFunction", "binGradientTable"}, {"binningVariables", JsonType::array()}};
  Event event;
  DialInputBuffer input;
  input.setInputValues({.3, -.2});
  const auto check = [&](auto& factory_){
    std::unique_ptr<DialBase> dial(factory_.makeDial(event));
    ASSERT_NE(nullptr, dial);
    factory_.updateTable(input); // values alone do not publish a gradient
    EXPECT_THROW(static_cast<void>(dial->evalGradients(input)), std::runtime_error);
    ASSERT_TRUE(dial->hasAnalyticGradient());
    factory_.updateTable(input);
    const auto gradient = dial->evalGradients(input);
    ASSERT_EQ(2U, gradient.size());
    for( int p = 0; p < 2; ++p ){
      const auto point = input.getInputBuffer();
      auto up = point, down = point;
      up[p] += 1E-6; down[p] -= 1E-6;
      input.setInputValues(up);
      factory_.updateTable(input);
      const double upper = dial->evalResponse(input);
      input.setInputValues(down);
      factory_.updateTable(input);
      const double lower = dial->evalResponse(input);
      EXPECT_NEAR(gradient[p], (upper - lower) / 2E-6, 1E-8);
      input.setInputValues(point);
    }
    ASSERT_TRUE(dial->hasAnalyticGradient());
    factory_.updateTable(input);
    EXPECT_EQ(gradient, dial->evalGradients(input));
  };
  TabulatedDialFactory tabulated(makeConfig(table));
  check(tabulated);
  auto missing = table;
  missing["gradientFunction"] = "missingGradientSymbol";
  EXPECT_THROW(TabulatedDialFactory{makeConfig(missing)}, std::runtime_error);
  table.erase("gradientFunction");
  TabulatedDialFactory valuesOnly(makeConfig(table));
  std::unique_ptr<DialBase> unsupported(valuesOnly.makeDial(event));
  EXPECT_FALSE(unsupported->hasAnalyticGradient());
  EXPECT_NO_THROW(valuesOnly.updateTable(input));
  table.erase("binningFunction");
  table.erase("binningVariables");
  table["weightFunction"] = "weightGradientTable";
  table["weightVariables"] = JsonType::array();
  table["gradientFunction"] = "differentiateGradientTable";
  KrigedDialFactory kriged(makeConfig(table));
  check(kriged);
  table.erase("gradientFunction");
  KrigedDialFactory krigedValuesOnly(makeConfig(table));
  std::unique_ptr<DialBase> unsupportedKriged(krigedValuesOnly.makeDial(event));
  EXPECT_FALSE(unsupportedKriged->hasAnalyticGradient());
  EXPECT_NO_THROW(krigedValuesOnly.updateTable(input));
}

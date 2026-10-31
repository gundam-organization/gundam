#include "DialCollection.h"
#include "DialResponseSupervisor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

TEST(DialResponseSupervisor, PreservesHardClamps){
  DialResponseSupervisor supervisor;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  for( double value : {-infinity, -2., 0., 0.5, 2., infinity} ){
    EXPECT_EQ(value, supervisor.process(value));
  }
  supervisor.setMinResponse(0.);
  supervisor.setMaxResponse(2.);
  for( double value : {-infinity, -2., 0., 0.5, 2., infinity} ){
    EXPECT_EQ(std::max(0., std::min(2., value)), supervisor.process(value));
  }
  EXPECT_TRUE(std::isnan(supervisor.process(nan)));
}

TEST(DialResponseSupervisor, CenteredValuesAndAsymmetricWidths){
  DialResponseSupervisor supervisor;
  supervisor.setMinResponse(0.);
  supervisor.setMaxResponse(2.);
  supervisor.setSmoothingWidths(0.4, 0.1);
  EXPECT_DOUBLE_EQ(0., supervisor.process(-0.4));
  EXPECT_NEAR(0.075, supervisor.process(0.), 1E-15);
  EXPECT_DOUBLE_EQ(0.4, supervisor.process(0.4));
  EXPECT_DOUBLE_EQ(1., supervisor.process(1.));
  EXPECT_DOUBLE_EQ(1.9, supervisor.process(1.9));
  EXPECT_NEAR(1.98125, supervisor.process(2.), 1E-15);
  EXPECT_DOUBLE_EQ(2., supervisor.process(2.1));
  EXPECT_DOUBLE_EQ(0., supervisor.process(-100.));
  EXPECT_DOUBLE_EQ(2., supervisor.process(100.));
  EXPECT_TRUE(std::isnan(supervisor.process(std::nan(""))));
}

TEST(DialResponseSupervisor, MonotoneWithoutSlopeOvershoot){
  DialResponseSupervisor supervisor;
  supervisor.setMinResponse(-0.5);
  supervisor.setMaxResponse(2.);
  supervisor.setSmoothingWidths(0.4, 0.1);
  double previous = supervisor.process(-1.);
  const double step = 0.0005;
  for( int index = 1; index <= 7000; ++index ){
    const double value = supervisor.process(-1. + index * step);
    EXPECT_GE(value, previous - 1E-14);
    EXPECT_LE(value - previous, step + 1E-14);
    EXPECT_GE(value, -0.5);
    EXPECT_LE(value, 2.);
    previous = value;
  }
}

TEST(DialResponseSupervisor, C2AtAllFourJunctions){
  DialResponseSupervisor supervisor;
  supervisor.setMinResponse(0.);
  supervisor.setMaxResponse(2.);
  supervisor.setSmoothingWidths(0.4, 0.1);
  for( const auto& boundary : {std::make_pair(-0.4, 0.), std::make_pair(0.4, 1.),
                              std::make_pair(1.9, 1.), std::make_pair(2.1, 0.)} ){
    const double x = boundary.first;
    const double h = 1E-5;
    const double value = supervisor.process(x);
    const double leftSlope = (value - supervisor.process(x - h)) / h;
    const double rightSlope = (supervisor.process(x + h) - value) / h;
    const double leftCurvature = (value - 2. * supervisor.process(x - h) + supervisor.process(x - 2. * h)) / (h * h);
    const double rightCurvature = (supervisor.process(x + 2. * h) - 2. * supervisor.process(x + h) + value) / (h * h);
    EXPECT_NEAR(boundary.second, leftSlope, 1E-7);
    EXPECT_NEAR(boundary.second, rightSlope, 1E-7);
    EXPECT_NEAR(0., leftCurvature, 0.004);
    EXPECT_NEAR(0., rightCurvature, 0.004);
  }
}

TEST(DialResponseSupervisor, AllowsTouchingTransitionsAndOneSidedLimits){
  DialResponseSupervisor supervisor;
  supervisor.setMinResponse(0.);
  supervisor.setMaxResponse(1.);
  supervisor.setSmoothingWidths(0.75, 0.25);
  EXPECT_DOUBLE_EQ(0.75, supervisor.process(0.75));
  EXPECT_NEAR(1., (supervisor.process(0.750001) - supervisor.process(0.749999)) / 0.000002, 1E-8);
  supervisor = DialResponseSupervisor();
  supervisor.setMaxResponse(1.);
  supervisor.setSmoothingWidths(std::nan(""), 0.25);
  EXPECT_DOUBLE_EQ(-5., supervisor.process(-5.));
  EXPECT_DOUBLE_EQ(1., supervisor.process(2.));
}

TEST(DialResponseSupervisor, RejectsInvalidWidths){
  DialResponseSupervisor supervisor;
  supervisor.setMinResponse(0.);
  supervisor.setMaxResponse(1.);
  for( double width : {0., -0.1, std::nan(""), std::numeric_limits<double>::infinity()} ){
    EXPECT_THROW(supervisor.setSmoothingWidths(width, 0.1), std::runtime_error);
    EXPECT_THROW(supervisor.setSmoothingWidths(0.1, width), std::runtime_error);
  }
  EXPECT_THROW(supervisor.setSmoothingWidths(0.75, 0.5), std::runtime_error);
}

TEST(DialResponseSmoothingConfig, ValidatesConfigurationBeforeLoading){
  auto configure = [](const JsonType& options_){
    JsonType config{{"dialType", "Graph"}, {"dialBranchData", "graph"}};
    config.update(options_);
    DialCollection collection(nullptr);
    collection.configure(ConfigReader(config));
  };
  const JsonType enabled{{"dialResponseLimits", {0., 2.}}, {"enableResponseSmoothing", true}};
  for( const JsonType& width : {JsonType(0.), JsonType(-0.1), JsonType("bad"), JsonType::array(),
                               JsonType{0.1}, JsonType{0.1, 0.}, JsonType{{"min", 0.1}},
                               JsonType{{"min", 0.1}, {"unexpected", 0.2}}, JsonType{1.5, 1.}} ){
    auto options = enabled;
    options["dialResponseSmoothingWidth"] = width;
    EXPECT_THROW(configure(options), std::runtime_error) << options;
  }
  EXPECT_THROW(configure(enabled), std::runtime_error);
  EXPECT_THROW(configure(JsonType{{"dialResponseLimits", {0., 2.}}, {"minDialResponse", 0.1}}), std::runtime_error);
  EXPECT_THROW(configure(JsonType{{"dialResponseLimits", {0., 2.}}, {"maxDialResponse", 3.}}), std::runtime_error);
  EXPECT_THROW(configure(JsonType{{"dialResponseLimits", {{"min", 2.}, {"max", 0.}}}}), std::runtime_error);
  EXPECT_NO_THROW(configure(JsonType{{"dialResponseLimits", {0., 2.}}, {"minDialResponse", 0.}, {"maxDialResponse", 2.}}));
  EXPECT_NO_THROW(configure(JsonType{{"dialResponseLimits", {0., 2.}}, {"dialResponseSmoothingWidth", {0.4, 0.1}}}));
  EXPECT_NO_THROW(configure(JsonType{{"dialResponseLimits", {{"max", 2.}}}, {"enableResponseSmoothing", true},
                                    {"dialResponseSmoothingWidth", {{"max", 0.1}}}}));
  EXPECT_NO_THROW(configure(JsonType{{"dialResponseLimits", {0., 2.}}, {"enableResponseSmoothing", true},
                                    {"dialResponseSmoothingWidth", {0.4, 0.1}}}));
}

TEST(DialResponseSmoothingConfig, LegacyLimitsWarnWithoutRejectingConfigurations){
  for( const std::string minKey : {"minDialResponse", "minimumSplineResponse"} ){
    DialCollection collection(nullptr);
    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(collection.configure(ConfigReader(JsonType{
        {"dialType", "Graph"}, {"dialBranchData", "graph"},
        {minKey, 0.2}, {"maxDialResponse", 1.8}})));
    const auto output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("key \"" + minKey + "\" is deprecated"));
    EXPECT_NE(std::string::npos, output.find("key \"maxDialResponse\" is deprecated"));
    EXPECT_NE(std::string::npos, output.find("dialResponseLimits"));
  }
}

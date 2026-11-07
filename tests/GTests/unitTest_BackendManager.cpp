#include "BackendManager.h"
#include "BackendFactory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>

TEST(BackendManager, ConfiguresEmptyAvailableBackendsAndRejectsLegacyOptions) {
  Backends::BackendManager manager;
  EXPECT_THROW(manager.configure(ConfigReader(JsonType{{"backendConfig", JsonType::object()}})), std::exception);
  EXPECT_THROW(manager.configure(ConfigReader(JsonType{{"options", JsonType::object()}})), std::exception);
  EXPECT_THROW(manager.configure(ConfigReader(JsonType{{"deviceIndex", 0}})), std::exception);
  manager.configure(ConfigReader(JsonType{{"options", JsonType::array({
      {{"name", "CPU"}, {"weightScheduling", "Fixed"}},
      {{"name", "CUDA"}, {"deviceIndex", -1}}
  })}}));
  const auto compiled = Backends::getCompiledBackendNames();
  if( std::find(compiled.begin(), compiled.end(), "CUDA") == compiled.end() ){
    EXPECT_NO_THROW(manager.initialize()); // Ignore all options for unavailable CUDA.
    EXPECT_EQ(manager.getActiveBackendName(), "");
    const auto names = manager.getAvailableBackendNames();
    EXPECT_NE(std::find(names.begin(), names.end(), "CPU"), names.end());
    EXPECT_EQ(std::find(names.begin(), names.end(), "CUDA"), names.end());
    EXPECT_THROW(manager.activate("CUDA"), std::exception);
    EXPECT_FALSE(manager.hasBackend());
    EXPECT_NO_THROW(manager.deactivateBackend());
  }
  Backends::BackendManager duplicate;
  duplicate.configure(ConfigReader(JsonType{{"options", JsonType::array({{{"name", "CPU"}}, {{"name", "CPU"}}})}}));
  EXPECT_THROW(duplicate.initialize(), std::exception);
}

TEST(BackendFactory, OnlyConstructsCompiledBackendsWithoutAllocatingADevice) {
  const auto names = Backends::getCompiledBackendNames();
  EXPECT_NE(std::find(names.begin(), names.end(), "CPU"), names.end());
  for( const auto* name : {"CPU", "CUDA", "MPS"} ){
    if( std::find(names.begin(), names.end(), name) == names.end() ){
      EXPECT_THROW((void)Backends::makeBackend(name), std::exception);
    }
    else {
      EXPECT_NE(Backends::makeBackend(name), nullptr);
    }
  }
}

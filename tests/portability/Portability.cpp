#include "DynamicLibrary.h"
#include "CalculateGraph.h"
#include "ExternalWeightDialFactory.h"
#include "DialInputBuffer.h"
#include "WindowsProcess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  void require(bool condition_, const char* message_){
    if( not condition_ ){ throw std::runtime_error(message_); }
  }

  void checkGraphs(){
    // Exercise searches on either side of powers of two, including extrapolation.
    for( int count = 2; count <= 129; ++count ){
      std::vector<double> data;
      for( int point = 0; point < count; ++point ){
        const double x = point*point + point;
        data.push_back(std::sin(point*0.7));
        data.push_back(x);
      }
      for( int point = 0; point < count; ++point ){
        for( const double displacement : {-0.25, 0., 0.25} ){
          const double x = data[2*point + 1] + displacement;
          int interval = 0;
          while( interval + 2 < count and x > data[2*(interval + 1) + 1] ){ ++interval; }
          const double fraction = (x - data[2*interval + 1]) / (data[2*(interval + 1) + 1] - data[2*interval + 1]);
          const double expected = data[2*interval] + fraction*(data[2*(interval + 1)] - data[2*interval]);
          require(std::abs(CalculateGraph(x, -1e9, 1e9, data.data(), static_cast<int>(data.size()))
                           - expected) < 1e-10, "Graph interpolation/extrapolation changed");
        }
      }
    }
  }

  void checkLibrary(const char* path_){
    auto* library = GundamUtils::DynamicLibrary::open(path_);
    if( library == nullptr ){ throw std::runtime_error(GundamUtils::DynamicLibrary::error()); }
    const auto function = reinterpret_cast<double(*)(double)>(
        GundamUtils::DynamicLibrary::symbol(library, "gundamPortabilityPlugin"));
    require(function != nullptr, "Plugin symbol not loaded");
    require(function(2) == 7, "Plugin evaluation failed");
    require(GundamUtils::DynamicLibrary::symbol(library, "missingSymbol") == nullptr, "Missing symbol accepted");
#if defined(_WIN32)
    SetLastError(ERROR_SUCCESS);
#endif
    require(not GundamUtils::DynamicLibrary::error().empty(), "Missing symbol diagnostic absent");
    require(GundamUtils::DynamicLibrary::error().empty(), "Loader diagnostic was not cleared");
    GundamUtils::DynamicLibrary::close(library);
  }

#if defined(_WIN32)
  void checkProcess(const char* executable_){
    const std::vector<std::string> arguments{"", "with spaces", "quote\"inside", "trailing\\", "\\\""};
    std::vector<std::string> command{executable_, "--child"};
    command.insert(command.end(), arguments.begin(), arguments.end());
    GundamUtils::WindowsProcess process{command, true};
    const std::string input{"request\n"};
    require(process.write(input.data(), input.size()) == input.size(), "Child stdin write failed");
    process.closeInput();
    std::string output;
    char buffer[256];
    while( const auto count = process.read(buffer, sizeof(buffer)) ){ output.append(buffer, count); }
    require(process.wait() == 7, "Child exit status changed");
    auto expected = JsonType(arguments).dump() + "\r\nrequest\r\nstderr\r\n";
    require(output == expected, "Child arguments or pipe contents changed");
  }
#endif

  void checkExternalWeight(){
    const char* python = std::getenv("GUNDAM_TEST_PYTHON");
    if( python == nullptr ){
      std::cout << "ExternalWeight integration skipped: set GUNDAM_TEST_PYTHON to Python with NumPy\n";
      return;
    }
    const auto directory = std::filesystem::temp_directory_path()
        / ("gundam-portability-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct DirectoryGuard {
      std::filesystem::path path;
      ~DirectoryGuard(){ std::error_code error; std::filesystem::remove_all(path, error); }
    } guard{directory};
    const auto script = directory / "weights with spaces.py";
    std::ofstream(script) << R"PY(
def configure(arguments):
    assert arguments == ['with spaces', 'quote"inside', 'trailing\\']

def run(command):
    if command['command'] == 'initialize':
        assert command['parameters'] == {}
    elif command['command'] == 'evaluate':
        command['weights'][:] = 2 * command['inputs']['x']
    return {'status': 'ok'}
)PY";
    JsonType config = {
        {"type", "PythonWorker"}, {"useBinnedWeights", true},
        {"binning", {{"binningDefinition", JsonType::array({{{"name", "x"}, {"edges", {0., 1., 2.}}}})}}},
        {"workerConfig", {{"pythonExecutable", python}, {"evalScript", script.string()},
                          {"scriptArgs", {"with spaces", "quote\"inside", "trailing\\"}}}}
    };
    ExternalWeightDialFactory factory;
    factory.configure(ConfigReader{config});
    factory.initialize();
    DialInputBuffer input;
    auto first = factory.makeBinnedDial(0);
    auto second = factory.makeBinnedDial(1);
    factory.updateWeights(input);
    require(first->evalResponse(input) == 1., "First shared-memory weight incorrect");
    require(second->evalResponse(input) == 3., "Second shared-memory weight incorrect");
    factory.updateWeights(input);
    require(second->evalResponse(input) == 3., "Worker reuse failed");
  }
}

int main(int argc_, char** argv_) try {
#if defined(_WIN32)
  if( argc_ > 1 and std::string(argv_[1]) == "--child" ){
    std::vector<std::string> arguments(argv_ + 2, argv_ + argc_);
    std::cout << JsonType(arguments).dump() << '\n';
    std::string line;
    std::getline(std::cin, line);
    std::cout << line << '\n' << std::flush;
    std::cerr << "stderr\n";
    return 7;
  }
  checkProcess(argv_[0]);
#endif
  require(argc_ == 2, "Expected the test plugin path");
  checkGraphs();
  checkLibrary(argv_[1]);
  checkExternalWeight();
  return 0;
}
catch( const std::exception& error ){
  std::cerr << error.what() << std::endl;
  return 1;
}

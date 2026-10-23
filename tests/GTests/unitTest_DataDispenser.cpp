#include "LikelihoodInterface.h"
#include "GundamGlobals.h"

#include <TFile.h>
#include <TGraph.h>
#include <TList.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
  class TemporaryRootFile {
  public:
    TemporaryRootFile(){
      TString name("gundam-reader-cleanup-");
      auto* file = gSystem->TempFileName(name, nullptr, ".root");
      if( file == nullptr ){ throw std::runtime_error("Could not create temporary ROOT file"); }
      std::fclose(file);
      path = name.Data();
    }
    ~TemporaryRootFile(){ std::remove(path.c_str()); }

    std::string path;
  };

  class RootReaderCleanup : public testing::TestWithParam<int> {
  protected:
    void SetUp() override {
      _previousThreadCount_ = GundamGlobals::getNbCpuThreads();
      GundamGlobals::setNumberOfThreads(GetParam());
    }
    void TearDown() override { GundamGlobals::setNumberOfThreads(_previousThreadCount_); }

  private:
    int _previousThreadCount_{1};
  };
}

TEST_P(RootReaderCleanup, ClosesMainAndFriendFilesBeforePropagation){
  TemporaryRootFile mainFile;
  TemporaryRootFile friendFile;
  {
    TFile file(mainFile.path.c_str(), "RECREATE");
    TGraph graph(4);
    for( int knot = 0; knot < 4; ++knot ){
      double x = knot - 1;
      graph.SetPoint(knot, x, 1. + .125 * x * x);
    }
    TGraph* graphPtr = &graph;
    double raw{0};
    TTree tree("events", "events");
    tree.Branch("raw", &raw, "raw/D");
    tree.Branch("graph", &graphPtr, 32000, 0);
    for( int entry = 0; entry < 4; ++entry ){
      raw = .25 + .5 * entry;
      tree.Fill();
    }
    tree.Write();
  }
  {
    TFile file(friendFile.path.c_str(), "RECREATE");
    double weight{0};
    double offset{0};
    TTree tree("weights", "weights");
    tree.Branch("weight", &weight, "weight/D");
    tree.Branch("offset", &offset, "offset/D");
    for( int entry = 0; entry < 4; ++entry ){
      weight = entry + 1;
      tree.Fill();
    }
    tree.Write();
  }

  auto config = JsonType::parse(R"({
    "jointProbabilityConfig": {"type": "LeastSquares", "lsqPoissonianApproximation": true},
    "datasetList": [{"name": "input", "model": {
      "tree": "events", "nominalWeightFormula": "weights.weight + 0",
      "variableDict": [{"name": "X", "expr": "raw + weights.offset"}]
    }}],
    "propagatorConfig": {
      "sampleSetConfig": {"sampleList": [{"name": "sample", "binning": {
        "binningDefinition": [{"name": "X", "edges": [0, 1, 2]}]
      }}]},
      "parametersManagerConfig": {"parameterSetList": [{
        "name": "shape", "isEnabled": true, "nominalStepSize": 0.1,
        "parameterDefinitions": [{"parameterName": "p", "priorValue": 0.5, "priorType": "Flat"}],
        "dialSetDefinitions": [{"dialType": "Spline", "dialBranchData": "graph",
                                "dialInputList": [{"name": "p"}]}]
      }]}
    }
  })");
  config["datasetList"][0]["model"]["filePathList"] = JsonType::array({{
    {"name", "main"},
    {"path", mainFile.path},
    {"friendList", JsonType::array({{{"name", "weights"}, {"path", friendFile.path + ":weights"}}})}
  }});

  LikelihoodInterface likelihood;
  likelihood.setForceAsimovData(true);
  likelihood.configure(ConfigReader(config));
  likelihood.initialize();

  // ROOT must no longer retain any loader handle, including the friend chains.
  EXPECT_EQ(nullptr, gROOT->GetListOfFiles()->FindObject(mainFile.path.c_str()));
  EXPECT_EQ(nullptr, gROOT->GetListOfFiles()->FindObject(friendFile.path.c_str()));

  auto& model = likelihood.getModelPropagator();
  auto& parameter = model.getParametersManager().getParameterSetsList()[0].getParameterList()[0];
  const auto& sample = model.getSampleSet().getSampleList()[0];
  ASSERT_EQ(4, sample.getEventList().size());
  for( double value : {0., .5, 1., .25, .5} ){
    parameter.setParameterValue(value);
    likelihood.propagateAndEvalLikelihood();
    const auto& bins = sample.getHistogram().getBinContentList();
    ASSERT_EQ(2, bins.size());
    double response = 1. + .125 * value * value;
    EXPECT_NEAR(3. * response, bins[0].sumWeights, 1e-12);
    EXPECT_NEAR(7. * response, bins[1].sumWeights, 1e-12);
  }
}

INSTANTIATE_TEST_SUITE_P(LoadingThreads, RootReaderCleanup, testing::Values(1, 2));

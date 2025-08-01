//
// Created by Clark McGrew 10/01/23
//

#ifndef GUNDAM_LIKELIHOOD_INTERFACE_H
#define GUNDAM_LIKELIHOOD_INTERFACE_H

#include "SamplePair.h"
#include "Propagator.h"
#include "ParameterSet.h"
#include "JointProbability.h"
#include "Propagator.h"
#include "EventTreeWriter.h"
#include "DatasetDefinition.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include <string>
#include <memory>


/// Evaluate the likelihood between data and MC.  The calculation is buffered
/// and and updated by the propagateAndEvalLikelihood() method.  The
/// likelihood value for the last calculation is accessible through
/// getLastLikelihood().
///
/// The "likelihood" for GUNDAM is based on the comparision between a "data"
/// and "expected" (i.e. MC) histogram.  The bin-by-bin comparisions are done
/// using the JointProbability class and are based on one of several LLH (Log
/// Likelihood) calculations (e.g. Barlow-Beeston, Icecube, Poissonian).
/// While the value is proportional to the LLH, it is more closely related to
/// the chi-square since we use -2*LLH.
class LikelihoodInterface : public JsonBaseClass  {

public:
#define ENUM_NAME DataType
#define ENUM_FIELDS \
  ENUM_FIELD( Asimov, 0 ) \
  ENUM_FIELD( Toy ) \
  ENUM_FIELD( RealData )
#include "GenericToolbox.MakeEnum.h"

  struct Buffer{
    double totalLikelihood{0};

    double statLikelihood{0};
    double penaltyLikelihood{0};

    void updateTotal(){ totalLikelihood = statLikelihood + penaltyLikelihood; }
    [[nodiscard]] bool isValid() const { return not ( std::isnan(totalLikelihood) or std::isinf(totalLikelihood) ); }
  };

protected:
  // called through JsonBaseClass::configure() and JsonBaseClass::initialize()
  void configureImpl() override;
  void initializeImpl() override;

public:

  // setters
  void setForceAsimovData(bool forceAsimovData_){ _forceAsimovData_ = forceAsimovData_; }
  void setDataType(const DataType& dataType_){ _dataType_ = dataType_; }
  void setToyParameterInjector(const JsonType& toyParameterInjector_){ _toyParameterInjector_ = toyParameterInjector_; }

  // const getters
  [[nodiscard]] auto getNbParameters() const{ return _nbParameters_; }
  [[nodiscard]] auto getNbSampleBins() const{ return _nbSampleBins_; }
  [[nodiscard]] auto getLastLikelihood() const{ return _buffer_.totalLikelihood; }
  [[nodiscard]] auto getLastStatLikelihood() const{ return _buffer_.statLikelihood; }
  [[nodiscard]] auto getLastPenaltyLikelihood() const{ return _buffer_.penaltyLikelihood; }
  [[nodiscard]] auto getJointProbabilityPtr() const { return _jointProbabilityPtr_.get(); }
  [[nodiscard]] auto& getModelPropagator() const{ return _modelPropagator_; }
  [[nodiscard]] auto& getDataPropagator() const{ return _dataPropagator_; }
  [[nodiscard]] auto& getPlotGenerator() const{ return _plotGenerator_; }
  [[nodiscard]] auto& getDatasetList() const { return _datasetList_; }
  [[nodiscard]] auto& getSamplePairList() const { return _samplePairList_; }
  [[nodiscard]] auto& getBuffer() const { return _buffer_; }
  [[nodiscard]] auto& getDataType() const { return _dataType_; }

  // mutable getters
  auto& getBuffer(){ return _buffer_; }
  auto& getModelPropagator(){ return _modelPropagator_; }
  auto& getDataPropagator(){ return _dataPropagator_; }
  auto& getPlotGenerator(){ return _plotGenerator_; }
  auto& getDatasetList(){ return _datasetList_; }
  auto& getSamplePairList(){ return _samplePairList_; }

  // mutable core
  void setCurrentParameterValuesAsPrior();
  void propagateAndEvalLikelihood();

  // core
  double evalLikelihood(std::future<bool>& propagation) const;
  double evalStatLikelihood(std::future<bool>& propagation) const;
  double evalPenaltyLikelihood() const;
  [[nodiscard]] double evalPenaltyLikelihood(const ParameterSet& parSet_) const;
  [[nodiscard]] double evalStatLikelihood(const SamplePair& samplePair_) const;
  [[nodiscard]] std::string getSummary() const;

  void writeEvents(const GenericToolbox::TFilePath& saveDir_) const;
  void writeEventRates(const GenericToolbox::TFilePath& saveDir_) const;

  // print
  void printBreakdowns() const;
  std::string getSampleBreakdownTable() const;


  void throwToyParameters(Propagator& propagator_);
  void throwStatErrors(Propagator& propagator_);

protected:
  void load();
  void loadModelPropagator();
  void loadDataPropagator();
  void buildSamplePairList();

  DataDispenser* getDataDispenser( DatasetDefinition& dataset_ );

private:
  // parameters
  bool _forceAsimovData_{false};
  bool _enableStatThrowInToys_{true};
  bool _gaussStatThrowInToys_{false};
  bool _enableEventMcThrow_{true};
  bool _applyInfinitePenaltyOnOutOfBoundPar_{false};
  DataType _dataType_{DataType::Asimov};
  JsonType _toyParameterInjector_{};

  // internals
  int _nbParameters_{0};
  int _nbSampleBins_{0};

  // multi-threading
  GenericToolbox::ParallelWorker _threadPool_{};

  /// user defined datasets
  std::vector<DatasetDefinition> _datasetList_;

  /// this is where model and data are kept to be compared
  Propagator _modelPropagator_{};
  Propagator _dataPropagator_{};

  EventTreeWriter _eventTreeWriter_{};
  PlotGenerator _plotGenerator_{};

  /// Statistical likelihood
  std::shared_ptr<JointProbability::JointProbabilityBase> _jointProbabilityPtr_{nullptr};

  /// Cache
  mutable Buffer _buffer_{};
  std::vector<SamplePair> _samplePairList_{};
};

#endif //  GUNDAM_LIKELIHOOD_INTERFACE_H

// An MIT Style License

// Copyright (c) 2022 Clark McGrew

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:

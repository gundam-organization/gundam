//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_PARAMETERSET_H
#define GUNDAM_PARAMETERSET_H

#include "Parameter.h"
#include "ParameterThrowerMarkHarz.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"

#include "TMatrixDSym.h"
#include "TVectorT.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TVectorT.h"
#include "TMatrixDSymEigen.h"

#include <vector>
#include <string>

/// Handle a set of parameters and dials that logically go together.  The
/// parameter values can be and which are correlated throw a user provided
/// covariance matrix that describes known correlations between the parameter
/// values.
class ParameterSet : public JsonBaseClass  {

protected:
  // called through JsonBaseClass::configure() and JsonBaseClass::initialize()
  void configureImpl() override;
  void initializeImpl() override;

public:
  // in src dependent
  static void muteLogger();
  static void unmuteLogger();

  // setters
  void setEnableDebugPrintout(bool enableDebugPrintout_){ _enableDebugPrintout_ = enableDebugPrintout_; }


  /// Process the input covariance matrix to make sure that fixed, free, and
  /// disabled parameters are detached from the from the covariance matrix.
  /// This also applies validity checks to the parameter set (e.g. make sure
  /// that eigendecomposed ParameterSets are not also applying parameter
  /// bounds).  The stripped covariance matrix is built, and the eigen
  /// decomposition is done (if requested).
  void processCovarianceMatrix();

  /// Define the type of validity that needs to be required by
  /// hasValidParameterValues.  This accepts a string with the possible values
  /// being:
  ///
  ///  "range" (default) -- Between the parameter minimum and maximum values.
  ///  "norange"         -- Do not require parameters in the valid range
  ///  "mirror"          -- Between the mirrored values (if parameter has
  ///                       mirroring).
  ///  "nomirror"        -- Do not require parameters in the mirrored range
  ///  "physical"        -- Only physically meaningful values.
  ///  "nophysical"      -- Do not require parameters in the physical range.
  ///
  /// Example: setParameterValidity("range,mirror,physical")
  void setValidity(const std::string& validity);
  void setValidity(int validity);

  // Getters
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool isScanEnabled() const{ return _isScanEnabled_; }
  [[nodiscard]] bool isEnablePca() const{ return _enablePca_; }
  [[nodiscard]] bool isEnableEigenDecomp() const{ return _enableEigenDecomp_; }
  [[nodiscard]] bool isEnabledThrowToyParameters() const{ return _enabledThrowToyParameters_; }
  [[nodiscard]] bool isMaskForToyGeneration() const{ return _maskForToyGeneration_; }
  [[nodiscard]] int getNbEnabledEigenParameters() const{ return _nbEnabledEigen_; }
  [[nodiscard]] double getPenaltyChi2Buffer() const{ return _penaltyChi2Buffer_; }
  [[nodiscard]] size_t getNbParameters() const{ return _parameterList_.size(); }
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const JsonType &getDialSetDefinitions() const{ return _dialSetDefinitions_; }
  [[nodiscard]] const TMatrixD* getInvertedEigenVectors() const{ return _eigenVectorsInv_.get(); }
  [[nodiscard]] const TMatrixD* getEigenVectors() const{ return _eigenVectors_.get(); }
  [[nodiscard]] const TVectorD* getDeltaVectorPtr() const{ return _deltaVectorPtr_.get(); }
  [[nodiscard]] const std::vector<JsonType>& getCustomParThrow() const{ return _customParThrow_; }
  [[nodiscard]] const std::shared_ptr<TMatrixDSym> &getPriorCovarianceMatrix() const { return _priorCovarianceMatrix_; }
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getInverseCovarianceMatrix() const{ return _inverseCovarianceMatrix_; }

  /// True if all of the enabled parameters have valid values.
  [[nodiscard]] bool isValid() const;

  /// Convenience method to check if a value will be valid for a particular
  /// parameter.
  [[nodiscard]] bool isValidParameterValue(const Parameter& p, double v) const;

  /// Get the vector of parameters for this parameter set in the real
  /// parameter space.  These parameters are not eigendecomposed.  WARNING:
  /// While the parameters are provided as a vector, elements must not be
  /// added or removed from the vector.  But, the value of the elements may be
  /// changed, so `getParameterList().front().setParameterValue(0)' is OK, but
  /// 'getParameterList().emplace_back(Parameter())' is NOT OK.
  [[nodiscard]] const std::vector<Parameter> &getParameterList() const{ return _parameterList_; }
  [[nodiscard]] std::vector<Parameter> &getParameterList(){ return _parameterList_; }

  /// Get the vector of parameters for this parameter set in the
  /// eigendecomposed basis.  WARNING: See warning for getParameterList().
  [[nodiscard]] const std::vector<Parameter> &getEigenParameterList() const{ return _eigenParameterList_; }
  [[nodiscard]] std::vector<Parameter> &getEigenParameterList(){ return _eigenParameterList_; }

  /// Get the vector of parameters for this parameter set that is applicable
  /// for the current stage of the fit.  This will either be the
  /// eigendecomposed parameters, or the parameters in the non-decomposed
  /// basis.  WARNING: See warning for getParameterList().
  [[nodiscard]] const std::vector<Parameter>& getEffectiveParameterList() const;
  [[nodiscard]] std::vector<Parameter>& getEffectiveParameterList();

  void updateDeltaVector() const;

  /// Set all the parameters to their prior values.
  void moveParametersToPrior();

  /// Set the parameter values based on a random throw with fluctuations
  /// determined by the striped covariance matrix.  If the first parameter,
  /// rethrowIfNotPhysical, is true, then the throw is retried until all of
  /// the parameters are within the physically allowed bounds.  if the second
  /// parameter, gain_, is set, it determines the variance of the thrown
  /// distribution relative to the stripped covariance matrix.  A value larger
  /// than one will increase the thrown variance.
  void throwParameters( bool rethrowIfNotPhysical_ = true, double gain_ = 1);

  /// Update the parameter values in the set based on the parameter values in
  /// the eigen decomposed basis.
  void propagateEigenToOriginal();

  /// Update the parameters in the eigen decomposed basis based on the
  /// parameter values in non-decomposed basis.
  void propagateOriginalToEigen();

  /// Pretty print a table summarizing the parameter set.
  [[nodiscard]] std::string getSummary() const;

  /// Build a JSON object to save the current values of the parameters.  The
  /// same object can be loaded using `injectParameterValues()` to restore the
  /// state and can be written to a file (See
  /// GenericToolbox::Json::toReadableString(JsonType) to make a "clean"
  /// string.
  [[nodiscard]] JsonType exportInjectorConfig() const;

  /// Set the parameter set parameter values from a JSON object that was written
  /// by hand or exportInjectorConfig.
  void injectParameterValues(const JsonType& config_);

  Parameter* getParameterPtr(const std::string& parName_);
  Parameter* getParameterPtrWithTitle(const std::string& parTitle_);

  // disable completely by wiping its members
  void nullify();

  // Normalize a value or range in units of the parameter StdDev
  static double toNormalizedParRange(double parRange, const Parameter& par);
  static double toNormalizedParValue(double parValue, const Parameter& par);

  // Convert a normalized value or range to a parameter value.
  static double toRealParValue(double normParValue, const Parameter& par);
  static double toRealParRange(double normParRange, const Parameter& par);

  /// A convenience function to check if a parameter is enabled, not free, and
  /// not fix.  This is true if the parameter should be in the stripped
  /// covariance matrix.
  static bool isValidCorrelatedParameter(const Parameter& par_);
  
  // print
  void printConfiguration() const;

protected:
  void readParameterDefinitionFile();
  void defineParameters();

  void setName(const std::string& name_){ _name_ = name_; }

private:
  // options
  bool _enableDebugPrintout_{false}; // used for printing out config reading

  // configuration
  bool _isEnabled_{false};
  bool _isScanEnabled_{true};
  bool _useMarkGenerator_{false};
  bool _useEigenDecompForThrows_{false};
  bool _printDialSetsSummary_{false};
  bool _printParametersSummary_{false};
  bool _releaseFixedParametersOnHesse_{false};
  bool _devUseParLimitsOnEigen_{false};
  int _nbParameterDefinition_{-1};
  int _maxNbEigenParameters_{-1};
  double _nominalStepSize_{std::nan("unset")};
  double _eigenSvdThreshold_{std::nan("unset")};
  double _maxEigenFraction_{1};
  std::string _name_{};
  std::string _parameterDefinitionFilePath_{};
  std::string _covarianceMatrixPath_{};
  std::string _parameterPriorValueListPath_{};
  std::string _parameterNameListPath_{};
  std::string _parameterLowerBoundsTVectorD_{};
  std::string _parameterUpperBoundsTVectorD_{};
  std::string _throwEnabledListPath_{};
  JsonType _parameterDefinitionConfig_{};
  JsonType _dialSetDefinitions_{};

  GenericToolbox::Range _globalParRange_{};
  GenericToolbox::Range _eigenParRange_{};

  // backward compatibility
  bool _maskForToyGeneration_{false};

  double _penaltyChi2Buffer_{std::nan("unset")};

  std::vector<JsonType> _enableOnlyParameters_{};
  std::vector<JsonType> _disableParameters_{};
  std::vector<JsonType> _customParThrow_{};

  // Eigen objects
  int _nbEnabledEigen_{0};
  bool _enablePca_{false};
  bool _enableEigenDecomp_{false};
  bool _allowEigenDecompWithBounds_{false};
  bool _useOnlyOneParameterPerEvent_{false};
  std::vector<Parameter> _eigenParameterList_{};
  std::shared_ptr<TMatrixDSymEigen> _eigenDecomp_{nullptr};

  // internals
  std::vector<Parameter> _parameterList_;

  // Toy throwing
  bool _enabledThrowToyParameters_{true};
  std::shared_ptr<TVectorD> _throwEnabledList_{nullptr};

  // Used for base swapping
  std::shared_ptr<TVectorD> _eigenValues_{nullptr};
  std::shared_ptr<TVectorD> _eigenValuesInv_{nullptr};
  std::shared_ptr<TMatrixD> _eigenVectors_{nullptr};
  std::shared_ptr<TMatrixD> _eigenVectorsInv_{nullptr};
  std::shared_ptr<TVectorD> _eigenParBuffer_{nullptr};
  std::shared_ptr<TVectorD> _originalParBuffer_{nullptr};
  std::shared_ptr<TMatrixD> _projectorMatrix_{nullptr};

  // original loaded from file
  std::shared_ptr<TMatrixDSym> _priorFullCovarianceMatrix_{nullptr};
  // matrices stripped from fixed/freed parameters
  std::shared_ptr<TMatrixDSym> _priorCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _inverseCovarianceMatrix_{nullptr}; // inverse matrix used for chi2

  std::shared_ptr<TVectorD>  _parameterPriorList_{nullptr};
  std::shared_ptr<TVectorD>  _parameterLowerBoundsList_{nullptr};
  std::shared_ptr<TVectorD>  _parameterUpperBoundsList_{nullptr};
  std::shared_ptr<TObjArray> _parameterNamesList_{nullptr};

  std::shared_ptr<TVectorD>  _deltaVectorPtr_{nullptr}; // difference from prior

  std::shared_ptr<TMatrixD> _choleskyMatrix_{nullptr};
  GenericToolbox::CorrelatedVariablesSampler _correlatedVariableThrower_{};
  std::shared_ptr<ParameterThrowerMarkHarz> _markHartzGen_{nullptr};

};


#endif //GUNDAM_PARAMETERSET_H

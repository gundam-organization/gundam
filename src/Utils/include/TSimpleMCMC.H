#ifndef TSimpleMCMC_H_SEEN
#define TSimpleMCMC_H_SEEN

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>

#include <TRandom.h>
#include <TFile.h>
#include <TTree.h>
#include <TMatrixD.h>
#include <TMatrixDSymEigen.h>
#include <TDecompChol.h>

typedef double Parameter;
typedef std::vector<Parameter> Vector;

// A typedef for how the parameters are written to a tree.
typedef std::vector<Parameter> SavedVector;

#ifndef MCMC_DEBUG_LEVEL
#define MCMC_DEBUG_LEVEL 2
#endif

#ifndef MCMC_DEBUG
#define MCMC_DEBUG(level) if (level <= (MCMC_DEBUG_LEVEL)) std::cout
#endif

#ifndef MCMC_ERROR
#define MCMC_ERROR (std::cout <<__FILE__<<":: "<<__LINE__<<": ")
#endif

class TProposeAdaptiveStep;

/// A ROOT based include-file-only templated class to run an MCMC.  The
/// resulting MCMC normally uses the Metropolis-Hastings algorithm with an
/// adaptive proposal function.  The UserLikelihood template argument must be
/// a class (or struct) which provides a method declared as:
///
///\code
/// struct ExampleLogLikelihood {
///    double operator() (const Vector& point);
/// }
///\endcode
///
/// The optional UserProposal template argument must be a struct (or class)
/// which provides (at least) these methods:
///
///\code
/// struct ExampleProposal {
///    void operator() (Vector& proposed,
///                     const Vector& previous,
///                     const double previousValue);
///    void InitializeState(const Vector& start, const double startValue);
///    bool RestoreState(const Vector& previous,
///                      const double previousValue,
///                      TTree* tree);
///    bool AttachState(TTree* tree);
///    bool SaveState();
///    bool StateSaved();
/// }
///\endcode
///
/// Where the methods must do the following:
///
///   - operator () : The output parameter "proposed" must be filled with a
///        new proposed point. The input parameter "previous" is the last
///        accepted point. The input parameter "previousValue" is the log
///        Likelihood at the previous accepted point.
///
///   - InitializeState() : Called before the first step.  THIS CAN BE A
///        NO-OP.
///
///   - RestoreState() : The input parameter "previous" is the last accepted
///        point. The input parameter "previousValue" is the log Likelihood at
///        the previous accepted point.  The input parameter "tree" is a tree
///        generated during the last run of the MCMC.  It will contain the
///        information saved by the last call to SaveState. The last (filled)
///        entry is the last saved information.  The branch definitions must
///        match with AttachState().  THIS CAN BE A NO-OP.
///
///   - AttachState() : Create the local branches in the tree.  The "tree"
///        parameter is the tree to get the new branches.  THIS CAN BE A
///        NO-OP.
///
///   - SaveState() : Fill the branch variables.  This should return true if
///        everything is filled (variables don't need to filled).  This does
///        not call TTree::Fill since that is done by the caller.  THIS CAN BE
///        A NO-OP.
///
///   - StateSaved() : Called to indicate that the state has been written to
///        the output tree.  This can be used to zero any branch variables.
///        THIS CAN BE A NO-OP.
///
/// This can be used in your root macros:
///
/// \code
/// class TDummyLogLikelihood {
/// public:
///     double operator()(const Vector& point)  const {
///         double logLikelihood = 0.0;
///         for (std::size_t i = 0; i < point.size(); ++i) {
///             logLikelihood += - 0.5*point[i]*point[i];
///         }
///         return logLikelihood;
///     }
/// };
///
/// void SimpleMCMC() {
///
///     TFile *outputFile = new TFile("simple-mcmc.root","recreate");
///     TTree *tree = new TTree("SimpleMCMC",
///                             "Tree of accepted points");
///
///     TSimpleMCMC<TDummyLogLikelihood> mcmc(tree);
///
///     // The next three lines are for example only and should almost never
///     // be used.  This is to show the syntax for controlling
///     // TProposeAdaptiveStep when you need to provide hints for the initial
///     // proposal funciton.  Don't just copy this blindly!  SetDim() needs
//      // to be used before SetGaussian and SetUniform.
///     mcmc.GetProposeStep().SetDim(5);
///     mcmc.GetProposeStep().SetGaussian(3,2.0); // Hint for proposal!
///     mcmc.GetProposeStep().SetUniform(4,-5,5); // If needed for special case.
///
///     // If you need provide hints about correlations between two dimensions.
///     mcmc.GetProposeStep().SetCorrelation(3,4,0.3)
///
///     Vector point(5);
///     mcmc.Start(point);  // Set initial conditions
///
///     // Example of burn-in without saving the steps.  You probably
///     // don't want to do this since you should study the behavior
///     // during burn-in.
///     for (int i=0; i<1000000; ++i) mcmc.Step(false);
///     mcmc.GetProposeStep().UpdateProposal();
///
///     // Run the chain and save the points.
///     for (int i=0; i<1000000; ++i) mcmc.Step();
///
///     tree->Write();
///     delete outputFile;
/// }
/// \endcode
///
/// If this macro is in a file named "SimpleMCMC.C", then you can run it using
///
/// \code
/// root -l -q SimpleMCMC.C+
/// \endcode
///
/// With ROOT5, the macro must be compiled using ACLIC.
///
/// The default class for UserProposal is TProposeAdaptiveStep which
/// implements an adaptive Metropolis-Hastings step.  It has several methods
/// that can be accessed using the GetProposeStep() method.  See above for an
/// example.
///
/// WARNING: This uses the random number generator provided by gRandom.  By
/// default that's "TRandom3", which is a good generator, but it is started
/// with a fixed value for the seed.  It is an extremely good idea to reset
/// the generator by adding
///
/// \code
/// gRandom = new TRandom3(0); // use the UUID to make a seed.
/// \endcode
///
/// \note Copyright 2017-2020 Clark McGrew (details at the end of the file).
/// The full distribution can be found at
/// https://github.com/ClarkMcGrew/root-simple-mcmc
///
template <typename UserLikelihood,
          typename UserProposal = TProposeAdaptiveStep>
class TSimpleMCMC {
public:

    /// Make the likelihood class available as TSimpleMCMC::LogLikelihood.
    typedef UserLikelihood LogLikelihood;

    /// Make the step proposal cclass available as TSimpleMCMC::ProposeStep.
    typedef UserProposal ProposeStep;

    /// Declare an object to run an MCMC.  The resulting MCMC normally uses
    /// the Metropolis-Hastings algorithm with an adaptive proposal function.
    /// This takes an optional pointer to a tree to save the accepted steps.
    /// If it is provided the constructor will add branch to the tree for the
    /// log likelihood and the parameters at the accepted step.  If the second
    /// optional parameter is true, then the proposed steps will also be added
    /// to the tree.
    TSimpleMCMC(TTree* tree = NULL, bool saveStep = false) : fTree(tree) {
        if (fTree) {
            MCMC_DEBUG(0) << "TSimpleMCMC: Adding branches to "
                          << fTree->GetName()
                          << std::endl;
            fTree->Branch("LogLikelihood",&fAcceptedLogLikelihood);
            fTree->Branch("TotalSteps", &fTotalSteps);
            fTree->Branch("Accepted",&fSaveAccepted);
            fTree->Branch("StepRMS",&fStepRMS);
            if (saveStep) {
                MCMC_DEBUG(0) << "TSimpleMCMC: Saving the trial steps."
                              << std::endl;
                fTree->Branch("Step",&fTrialStep);
            }
        }
        fLogLikelihoodCount = 0;
        fTotalSteps = 0;
        fProposeStep.AttachState(fTree);
        fStepRMS = 0.0;
        fStepRMSTrials = 0;
        fStepRMSWindow = 1000;
    }

    /// Get a reference to the object that will propose the step.  The
    /// ProposeStep object is constructed by the TSimpleMCMC template.  You
    /// can get write access to that object with this method, and can access
    /// any of your declared methods.  Because this is a template, ProposeStep
    /// is actually just a typedef for your class.
    ProposeStep& GetProposeStep() {return fProposeStep;}

    /// Get a reference to the likelihood calculation object.  The
    /// LogLikelihood object is constructed by the TSimpleMCMC template and
    /// is the class that you handed to the template.  You have write access
    /// to that object with this method, and can access any of your declared
    /// methods.  Because this is a template, TSimpleMCMC::LogLikelihood is
    /// actually just a typedef for your class.
    LogLikelihood& GetLogLikelihood() {return fLogLikelihood;}

    /// Get the number of times the log likelihood has been called.
    int GetLogLikelihoodCount() {return fLogLikelihoodCount;}

    /// Set the starting point for the mcmc.  If the optional argument is
    /// true, then the point will be saved to the output.
    void Start(Vector start, bool save=true) {
        fProposed.resize(start.size());
        std::copy(start.begin(), start.end(), fProposed.begin());

        fAccepted.resize(start.size());
        std::copy(start.begin(), start.end(), fAccepted.begin());

        fSaveAccepted.reserve(fAccepted.size());

        fTrialStep.resize(start.size());
        std::copy(start.begin(), start.end(), fTrialStep.begin());

        fProposedLogLikelihood = GetLogLikelihoodValue(fProposed);
        fAcceptedLogLikelihood = fProposedLogLikelihood;

        fProposeStep.InitializeState(fAccepted,fAcceptedLogLikelihood);

        if (save) SaveStep(false);
    }

    /// Restore the state from a previous chain.  If randomize is true, then
    /// this will find a random position in the existing tree to start from.
    /// Set "randomize" to true with care, and only if you understand the
    /// danger!!!
    void Restore(TTree* tree, bool randomize = false) {
        // A place to get total number of steps that have been tried for any
        // reason.  This includes both successes and failures.
        int getTotalSteps;

        /// A place to get the last accepted point.  This will be the same as
        /// the proposed point if the last step was accepted.
        SavedVector getAccepted;
        SavedVector* addrAccepted = &getAccepted;

        /// A place to get the likelihood at the last accepted point.
        double getAcceptedLogLikelihood;

        /// A place to get the step rms at the last accepted point
        double getStepRMS;

        MCMC_DEBUG(0) << "Restore the state" << std::endl;

        tree->SetBranchAddress("LogLikelihood",&getAcceptedLogLikelihood);
        tree->SetBranchAddress("TotalSteps", &getTotalSteps);
        tree->SetBranchAddress("Accepted",&addrAccepted);
        tree->SetBranchAddress("StepRMS",&getStepRMS);

        fTotalSteps = -1;
        fAcceptedLogLikelihood = 0;
        int elem = tree->GetEntries();
        while (elem > 1) {
            -- elem;
            tree->GetEntry(elem);
            if (fTotalSteps > 0) {
                double newP = std::exp(getAcceptedLogLikelihood);
                double oldP = std::exp(fAcceptedLogLikelihood);
                double r = gRandom->Uniform();
                // Reject the new point based on the ratio of probability with
                // respect to the last accepted point.
                if (newP < (newP+oldP)*r) continue;
            }

            fTotalSteps = getTotalSteps;
            fAcceptedLogLikelihood = getAcceptedLogLikelihood;
            fStepRMS = getStepRMS;
            fAccepted.resize(getAccepted.size());
            std::copy(getAccepted.begin(), getAccepted.end(),
                      fAccepted.begin());
            fProposed.resize(getAccepted.size());
            std::copy(getAccepted.begin(), getAccepted.end(),
                      fProposed.begin());
            fTrialStep.resize(getAccepted.size());
            std::copy(getAccepted.begin(), getAccepted.end(),
                      fTrialStep.begin());
            if (!randomize) break;
        }

        fProposedLogLikelihood = GetLogLikelihoodValue(fProposed);
        double delta = fProposedLogLikelihood - fAcceptedLogLikelihood;
        if (std::abs(delta) > 1E-4) {
            MCMC_ERROR << "Calculated likelihood doesn't match saved likelihood"
                       << std::endl;
            MCMC_ERROR << "Saved:      " << fAcceptedLogLikelihood
                       << std::endl;
            MCMC_ERROR << "Calculated: " << fProposedLogLikelihood
                       << std::endl;
            fAcceptedLogLikelihood = fProposedLogLikelihood;
        }

        tree->SetBranchAddress("LogLikelihood",NULL);
        tree->SetBranchAddress("TotalSteps", NULL);
        tree->SetBranchAddress("Accepted",NULL);

        fProposeStep.RestoreState(fAccepted,fAcceptedLogLikelihood,tree);
    }

    /// Take a step.  This returns true if a new point has been accepted, and
    /// false if we stay at the old point.  If save is true, then the points
    /// are saved to the output.  The first parameter, "save", can be set to
    /// false to save file space but should almost always be true.  The second
    /// parameter, "metropolis", can be used to help debug a complicated
    /// likelihood, but should almost always be zero (and not used).
    ///
    /// Defined values for "metropolis":
    ///
    ///     * 0 : Do a normal metropolis step.
    ///
    ///     * 1 : Only take "down hill" steps.  This turns the algorythm into
    ///           a very slow way to maximize the likelihood..
    ///
    ///     * 2 : Accept every step.  This can be used to scan the likelihood.
    ///
    bool Step(bool save=true, int metropolis=0) {
        if (fProposed.empty() || fAccepted.empty()) {
            MCMC_ERROR << "Must initialize starting point" << std::endl;
            throw std::invalid_argument("Uninitialized starting point");
        }

        ++fTotalSteps;

        fProposeStep(fProposed,fAccepted,fAcceptedLogLikelihood);

        // Make sure that the last accepted is in the "save" accepted
        // location.  This is needed uncase the user did something evil, like
        // zero the save location (to save memory).
        if (fSaveAccepted.size() != fAccepted.size()) {
            fSaveAccepted.resize(fAccepted.size());
            std::copy(fAccepted.begin(), fAccepted.end(),
                      fSaveAccepted.begin());
        }

        // Only cache the trial step when it's being saved in the tree, or the
        // step RMS is being calculated.
        if (save || fStepRMSWindow > 0) {
            double sqr = 0.0;
            for (std::size_t i = 0; i < fProposed.size(); ++i) {
                fTrialStep[i] = fProposed[i] - fAccepted[i];
                sqr += fTrialStep[i] * fTrialStep[i];
            }
            if (fStepRMSWindow > 0) {
                // Update the step RMS over the last "fStepRMSWindow" trials.
                double ms = fStepRMS*fStepRMS;
                ms *= fStepRMSTrials;
                ms += sqr;
                ms /= fStepRMSTrials + 1.0;
                fStepRMSTrials = std::min(fStepRMSWindow,fStepRMSTrials+1);
                fStepRMS = std::sqrt(ms);
            }
        }

        // Find the log likelihood at the new step.  The old likelihood has
        // been cached.
        fProposedLogLikelihood = GetLogLikelihoodValue(fProposed);

        /// This is when all steps should be accepted.  This can be used to
        /// force calculation of the likelihood at a cloud of points.
        if (metropolis == 2) {
            MCMC_DEBUG(0) << "Scanning step" << std::endl;
            // Always keep the new step.
            std::copy(fProposed.begin(), fProposed.end(), fAccepted.begin());
            // Save a copy for output
            fSaveAccepted.resize(fAccepted.size());
            std::copy(fProposed.begin(), fProposed.end(),
                      fSaveAccepted.begin());
            fAcceptedLogLikelihood = fProposedLogLikelihood;
            // Check to see if it gest written.
            if (save) SaveStep(false);
            return true;
        }

        // Extremely negative log likelihoods are treated as being zero. In
        // practice, zero probability is flagged by a non finite value
        // (e.g. inf, -inf, or nan), or by a log likelihood of less than
        // -1E+30.  That's a probability of less than 10**(-4E+29).
        if (!std::isfinite(fProposedLogLikelihood)
            || (fProposedLogLikelihood < -0.999999E+30)) {
            if (save) SaveStep(false);
            return false;
        }

        // Decide if the proposed step should be taken. When delta is
        // negative, the proposed point is less probable than the accepted
        // point.
        double delta = fProposedLogLikelihood - fAcceptedLogLikelihood;
        if (delta < 0.0 ) {
            // If the metropolis method parameter is false, then don't apply
            // the Metropolis algorithm and only take steps that increase the
            // probability.  Setting metropolis to false turns this into an
            // inefficient maximum likelihood fitter and can help test
            // complicated likelihoods, but it should almost always be true.
            if (metropolis == 1) return false;

            // The new point is less probable than the accepted point.  Apply
            // the Metropolis-Hastings condition to see if the step should be
            // made.  This is using the Metropolis special case of the
            // Metropolis-Hastings algorithm and depends on the proposal being
            // a symetric function (like a Gaussian).
            double trial = std::log(gRandom->Uniform());
            if (delta < trial) {
                // The new step should be rejected, so save the old step.
                // This depends on IEEE error handling so that std::log(0.0)
                // is -inf which is always less than delta.
                if (save) SaveStep(false);
                return false;
            }
        }

        // If this is being used as an (inefficient) minimization algorithm,
        // print the new step when one is found.
        if (metropolis == 1) {
            static int localCount = 0;
            MCMC_DEBUG(1) << localCount
                          << " L: "   << fProposedLogLikelihood;
            const std::size_t elems = 4;
            for (std::size_t i = 0;
                 i < std::min(elems,fProposed.size()); ++i) {
                MCMC_DEBUG(1) << " [" << i << "]=" << fProposed[i];
            }
            if (elems < fProposed.size()) {
                MCMC_DEBUG(1) << " [..." << fProposed.size() << "]";
            }
            MCMC_DEBUG(1) << std::endl;
            ++localCount;
        }

        // We're keeping a new step.
        fAcceptedLogLikelihood = fProposedLogLikelihood;

        // Save it for internal usage
        std::copy(fProposed.begin(), fProposed.end(), fAccepted.begin());

        // Save a copy for output
        fSaveAccepted.resize(fAccepted.size());
        std::copy(fProposed.begin(), fProposed.end(), fSaveAccepted.begin());

        // Save the information to the output tree.
        if (save) SaveStep(false);
        return true;
    }

    /// Get the likelihood at the most recently accepted point.
    double GetAcceptedLogLikelihood() const {return fAcceptedLogLikelihood;}

    /// Get the most recently accepted point.
    const Vector& GetAccepted() const {return fAccepted;}

    /// Get the likelihood at the most recently proposed point.
    double GetProposedLogLikelihood() const {return fProposedLogLikelihood;}

    /// Get the recent step length RMS.
    double GetStepRMS() const { return fStepRMS;}

    /// Set the number of trials to use when calculating the step RMS.
    void SetStepRMSWindow(int n) {fStepRMSWindow = n;}

    /// Get the most recently proposed point.
    const Vector& GetProposed() const {return fProposed;}

    /// Clear the accepted position data that will be saved to the file.  This
    /// can be used to help reduce the size of the output file.  This only
    /// affect the data saved in the output file, and does not accept the
    /// actual chain.
    void ClearSavedAccepted() {fSaveAccepted.clear();}

    /// If possible, save the step.  The only time that user code needs to
    /// call this method is after the last step of the chain, and even then
    /// that is only required if the chain will be extended in another run.
    /// The forceSave parameter is a cheap way to flag if this is called by
    /// the user, or directly by TSimpleMCMC.  TSimpleMCMC always uses
    /// SaveStep(false), and users should (usually) use SaveStep().
    void SaveStep(bool forceSave=true) {
        fProposeStep.SaveState(forceSave);
        if (fTree) fTree->Fill();
        fProposeStep.StateSaved();
    }

protected:

    /// A wrapper around the call to the likelihood.  The main purpose is to
    /// count the number of times the likelihood is called.
    double GetLogLikelihoodValue(const Vector& point) {
        ++fLogLikelihoodCount;
        return fLogLikelihood(point);
    }

    /// A class (called as a functor) to calculate the likelhood.
    LogLikelihood fLogLikelihood;

    /// A class (called as a functor) to propose the next step.
    ProposeStep fProposeStep;

    /// A TTree to save the accepted points.
    TTree* fTree;

    // The total number of steps that have been tried for any reason.  This
    // includes both successes and failures.
    int fTotalSteps;

    /// The number of times the likelihood has been calculated.
    int fLogLikelihoodCount;

    /// The last accepted point.  This will be the same as the proposed point
    /// if the last step was accepted.
    Vector fAccepted;

    /// A copy of the accepted points to be attached to a tree and save the
    /// accepted points.
    SavedVector fSaveAccepted;

    /// The likelihood at the last accepted pont.
    double fAcceptedLogLikelihood;

    /// The trial step.  The difference between the previous point and the new
    /// proposed point.
    Vector fTrialStep;

    /// The proposed point for the most recent step (This may be the same as
    /// the accepted point).
    Vector fProposed;

    /// The RMS of the proposed step length.  This will be averaged over
    /// recent steps.
    double fStepRMS;

    /// The number of trials currently used to estimate the step length RMS
    int fStepRMSTrials;

    /// The maximum number of trials to be used to estimiate the step length RMS
    int fStepRMSWindow;

    /// The likelihood at the last proposed point.
    double fProposedLogLikelihood;
};

// This is a very simple example of a step proposal class.  It's not actually
// used for anything, but implements a Metropolis-Hastings step.  The proposal
// is the new point to be tried.  The current is the last successful step, and
// the value is the log likelihood of the last successful step.  The step is
// taken relative to the current and the value is ignored.
struct TProposeSimpleStep {
    TProposeSimpleStep(): fSigma(-1.0) {}

    void operator ()(Vector& proposal,
                     const Vector& current,
                     const double value) const {
        double sigma = fSigma;

        // No width was provided, so make a bogus guess at a reasonable width;
        if (sigma < 0.0) sigma = std::sqrt(1.0/proposal.size());

        // Make the proposal.
        for (std::size_t i = 0; i < proposal.size(); ++i) {
            proposal[i] = current[i] + gRandom->Gaus(0.0,sigma);
        }
    }

    double fSigma;

    void InitializeState(const Vector& current, const double value) {}
    bool RestoreState(const Vector& current, const double value,
                      TTree* tree) {return false;}
    bool AttachState(TTree* tree) {return false;}
    bool SaveState(bool force) {return false;}
    bool StateSaved() {return false;}
    void SetDim(int dim) {};
    void UpdateProposal() {};

};

/// A default for the class to propose the next step.  This implements an
/// adaptive Metropolis-Hastings proposal.  It starts with a guess at the
/// covariance of the posterior, and the updates the estimated posterior.  At
/// user defined intervals (normally about the dimensionality squared), the
/// estimated covariance is updated using the current state of the Markov
/// chain. (Notice that this means it's not really a Markov Chain!  You need
/// to check the ergodcity, but it's almost always OK.  Do not update too
/// frequently) It is valid long as the posterior is not to badly behaved
/// (e.g. more or less Gaussian).  If the posterior is very non Gaussian it
/// can become much less efficient.
///
/// Methods:
///
class TProposeAdaptiveStep {
public:
    TProposeAdaptiveStep() :
        fLastValue(0.0),
        fCentralPointTrials(0.0), fCovarianceTrials(0.0),
        fCovarianceDeweight(0.5), fCovarianceWindow(-1),
        fTrials(0), fSuccesses(0), fNextUpdate(-1),
        fAcceptance(0.0), fAcceptanceTrials(0), fAcceptanceDeweight(0.5),
        fAcceptanceWindow(-1), fAcceptanceRigidity(2.0),
        fTargetAcceptance(-1), fSigma(0.0), fStateInitialized(false),
        fScanDimension(-1) {
        fMaxCorrelation = std::numeric_limits<Parameter>::epsilon();
        fMaxCorrelation = 1.0 - std::sqrt(fMaxCorrelation);
    }

    /// Take a proposed trial point to fill, the current point, and the
    /// likelihood at the current point.
    void operator ()(Vector& proposal,
                     const Vector& current,
                     const double value) {
        if (proposal.size() != current.size()) {
            // Apply a sanity check.  This MUST be true.
            MCMC_ERROR << "Proposal and current vectors must be same size."
                       << std::endl;
            throw std::logic_error("Proposal and current length mismatch");
        }

        /// If the "scan" variable is set, then only change that variable.
        /// This can be used to check the value of the likelihood versus the
        /// scan variable.  This does not update the proposal internal
        /// information.  It will scan the variable around the current central
        /// point.
        int scan = fScanDimension;
        if (scan >= 0 && scan < (int) proposal.size()) {
            std::copy(current.begin(), current.end(), proposal.begin());
            // Make the scan proposal.
            if (fProposalType[scan].type == 1) {
                // Make a uniform proposal.
                proposal[scan] = gRandom->Uniform(fProposalType[scan].param1,
                                                  fProposalType[scan].param2);
                return;
            }
            // Make a Gaussian proposal around the initial value.
            double sigma = 1.0;
            if (fProposalType[scan].param1>0) {
                sigma = std::sqrt(fProposalType[scan].param1);
            }
            double r = gRandom->Gaus(fCentralPoint[scan],sigma);
            proposal[scan] = r;
            MCMC_DEBUG(0) << "propose " << r << " for " << scan << std::endl;
            return;
        }

        UpdateState(current,value);

        std::copy(current.begin(), current.end(), proposal.begin());

        // Make the proposal.
        for (std::size_t i = 0; i < proposal.size(); ++i) {
            if (fProposalType[i].type == 1) {
                // Make a uniform proposal.
                proposal[i] = gRandom->Uniform(fProposalType[i].param1,
                                               fProposalType[i].param2);
                continue;
            }
            // Make a Gaussian Proposal (with the latest estimate of the
            // covariance).
            double r = gRandom->Gaus(0.0,1.0);
            for (std::size_t j = 0; j < proposal.size(); ++j) {
                if (fProposalType[j].type == 1) continue;
                proposal[j] += fSigma*r*fDecomposition(i,j);
            }
        }
    }

    /// Get (set) the current estimated center of the point cloud.  The
    /// estimated central point is a running average of the center of the
    /// point cloud and is used in the calculation of the current point cloud
    /// covariance.  A case where it might be set is when there is prior
    /// information about the expected central value.
    const Vector& GetEstimatedCenter() const {return fCentralPoint;}
    bool SetEstimatedCenter(const Vector& v) {
        if (fCentralPoint.size() != v.size()) return false;
        std::copy(v.begin(), v.end(), fCentralPoint.begin());
        return true;
    }

    /// Get the number of trials that are currently included in the central
    /// point estimation.
    double GetEstimatedCenterTrials() const {return fCentralPointTrials;}

    /// Set the effective number of trials that should be associated with the
    /// current central point value.  This is mostly useful when you know a
    /// good initial value for the central point (which is set using
    /// SetEstimatedCenter().
    void SetEstimatedCenterTrials(double v) {
        fCentralPointTrials = v;
        MCMC_DEBUG(1) << "Set estimated center trials to "
                      << fCentralPointTrials
                      << std::endl;
        if (fCentralPointTrials > fCovarianceWindow) {
            MCMC_DEBUG(0) << "Number of effective covariance trials"
                          << " (" << fCentralPointTrials << ")"
                          << " is set larger than window"
                          << " (" << fCovarianceWindow << ")"
                          << std::endl;
        }
    }

    /// Get the number of steps tried in the current chain.
    int GetTrials() const {return fTrials;}

    /// Get the number of successful steps in the current chain.
    int GetSuccesses() const {return fSuccesses;}

    /// Get the current step size in standard deviations of the current
    /// estimate of the posterior covariance.  The step size is adjusted to
    /// have an acceptance (approximately) equal to the target acceptance.
    double GetSigma() const {return fSigma;}

    /// Get the current acceptance (averaged over the acceptance window).  The
    /// step size will be adjusted so that this asymtotically approaches the
    /// target acceptance.  The usual target acceptance is about 23% (when
    /// there are more than 5 dimensions.
    double GetAcceptance() const {return fAcceptance;}
    double GetAcceptanceTrials() const {return fAcceptanceTrials;}

    /// Set the number of dimensions in the proposal.  This must match the
    /// dimensionality of the likelihood being use.
    void SetDim(int dim) {
        if (!fLastPoint.empty()) {
            // Apply a sanity check.  This MUST be true.
            MCMC_ERROR << "Dimensionality has already been set."
                       << std::endl;
            return;
        }
        fLastPoint.resize(dim);
        fProposalType.resize(dim);
    }
    int GetDim() const {return fLastPoint.size();}

    /// Set the dimension to be scanned (only use for debugging the likelihood)
    void SetScanDimension(int dim) {
        if (fLastPoint.empty()) {
            // Apply a sanity check.  This MUST NOT be true.
            MCMC_ERROR << "Dimensionality is not set."
                       << std::endl;
            return;
        }
        if (dim < 0) fScanDimension = -1;
        else if (dim >= (int) fLastPoint.size()) fScanDimension = -1;
        else fScanDimension = dim;
    }

    /// Set the proposal function for a particular dimension to be uniform.
    void SetUniform(int dim, double minimum, double maximum) {
        if (dim < 0 || (std::size_t) dim >= fProposalType.size()) {
            MCMC_ERROR << "Dimension " << dim << " is out of range."
                       << " 0 to " << fProposalType.size()
                       << std::endl;
            return;
        }
        MCMC_DEBUG(2) << "Overriding proposal for dimension " << dim
                      << " to be uniform between "
                      << "[" << minimum
                      << ", " << maximum << "]."
                      << std::endl;
        fProposalType[dim].type = 1;
        fProposalType[dim].param1 = minimum;
        fProposalType[dim].param2 = maximum;
    }

    /// Set the proposal function for a particular dimension to be Gaussian.
    /// This is the default, so it's only needed if you need to give a hint
    /// about the width of the proposal (for instance the dimension is very
    /// wide, or very narrow).  If you don't provide this, then the initial
    /// step starts out with a unit variance (i.e. sigma = 1.0)
    void SetGaussian(int dim, double sigma) {
        if (dim < 0 || (std::size_t) dim >= fProposalType.size()) {
            MCMC_ERROR << "Dimension " << dim << " is out of range."
                       << std::endl;
            return;
        }
        MCMC_DEBUG(2) << "Overriding proposal for dimension " << dim
                      << " to be Gaussian with "
                      << sigma << " sigma."
                      << std::endl;
        fProposalType[dim].type = 0;
        fProposalType[dim].param1 = sigma*sigma;
    }

    /// Reset the correlations (removes all prior correlations).  This should
    /// be used when the correlations between the parameters will be changed.
    /// If it's not called, you will always get the last values of the
    /// correlations that were set, but an internal vector will "grow without
    /// end".
    void ResetCorrelations() {fCorrelations.clear();}

    /// Set the correlation between two dimensions of the proposal function.
    /// The initial proposal function assumes that all of the dimensions are
    /// uncorrelated.  The optimal proposal will be the same as the posterior
    /// (which you don't know), but if you know something about the
    /// correlations of the posterior, you can provide hints using this
    /// method.  The provided value is the correlation coefficient between the
    /// two dimensions.
    void SetCorrelation(int dim1, int dim2, double correlation) {
        if (dim1 == dim2) {
            MCMC_ERROR << " Dimensions must be different for correlations"
                       << " (" << dim1 << "," << dim2 << ") = "
                       << correlation
                       << std::endl;
            return;
        }
        if (correlation < -fMaxCorrelation) {
            MCMC_ERROR << "Correlation is out of valid range: "
                       << correlation
                       << std::endl;
            correlation = -fMaxCorrelation;
        }
        if (correlation > fMaxCorrelation) {
            MCMC_ERROR << "Correlation is out of valid range: "
                       << correlation
                       << std::endl;
            correlation = fMaxCorrelation;
        }
        fCorrelations.push_back(CorrelationRecord(dim1,dim2,correlation));
    }

    /// This is the maximum allowed correlation between dimensions of the
    /// step proposal.  It is used to manage numeric error accumulatino, but
    /// could also be set based on the known properties of the posterior.
    void SetMaximumCorrelation(double c) {fMaxCorrelation = c;}

    /// Set (get) the window over which to estimate the covariance.  This can
    /// normally be very large (and there is a reasonable default), but it
    /// might need to be smaller for some pathelogical distributions.
    void SetCovarianceWindow(int w) {fCovarianceWindow = w;}
    double GetCovarianceWindow() const {return fCovarianceWindow;}

    /// Set the amount that the old covariance should be deweighted when there
    /// is an update.  This lets the new information have more effect on the
    /// covariance, and is mostly useful during burn-in.  After burn-in is
    /// finished the deweighing can (should?) be set to zero.
    ///
    /// The deweighting should be between 0.0 and 1.0.  A zero (0.0)
    /// deweighting means that the existing covariance is directly used in the
    /// average.  A deweighting of one (1.0) means that the existing
    /// covariance is NOT used in the running average.  Larger deweighting
    /// means that the newly accepted points will have more effect.
    void SetCovarianceUpdateDeweighting(double d) {fCovarianceDeweight = d;}

    /// Get the number of trials that are currently included in the covariance
    /// calculation.
    double GetCovarianceTrials() const {return fCovarianceTrials;}

    /// Set the effective number of trials that should be associated with the
    /// current covariance value.  This is mostly useful when you know a good
    /// initial value for the step proposal covariance.
    void SetCovarianceTrials(double v) {
        fCovarianceTrials = v;
        MCMC_DEBUG(1) << "Set covariance trials to " << fCovarianceTrials
                      << std::endl;
        if (fCovarianceTrials > fCovarianceWindow) {
            MCMC_DEBUG(0) << "Number of effective covariance trials"
                          << " (" << fCovarianceTrials << ")"
                          << " is set larger than window"
                          << " (" << fCovarianceWindow << ")"
                          << std::endl;
        }
    }

    /// Get the trace of the covariance.
    double GetCovarianceTrace() const {
        double trace = 0.0;
        for (std::size_t i = 0; i<fLastPoint.size(); ++i) {
            trace += fCurrentCov(i,i);
        }
        return trace;
    }

    /// Set (get) the target acceptance rate.  The literature proposes values
    /// between 44% (for one dimension) down to an upper bound of 23.4% above
    /// about five dimensions.  See https://doi.org/10.1016/j.spa.2007.12.005
    /// "Optimal acceptance rates for Metropolis algorithms: Moving beyond
    /// 0.234" (Stochastic Processes and their Applications, Volume 118, Issue
    /// 12, 2198-2222, 2008) for a full technical discussion.  My reading of
    /// the paper is the acceptance rate should "usually" be about 23%, but
    /// sometimes it should be smaller.
    void SetTargetAcceptance(double a) {fTargetAcceptance = a;}
    double GetTargetAcceptance() const {return fTargetAcceptance;}

    /// Set (get) the number of trials used to calculate the
    /// acceptance.
    void SetAcceptanceWindow(double a) {fAcceptanceWindow = a;}
    double GetAcceptanceWindow() const {return fAcceptanceWindow;}

    /// Set the amount that the old covariance should be deweighted when there
    /// is an update.  See SetCovarianceUpdateDeweighting for more details.
    void SetAcceptanceUpdateDeweighting(double d) {fAcceptanceDeweight = d;}

    /// Set (get) the number of trials until the next proposal update.  This
    /// grows with each successfull update, but can be overridden by this
    /// method.
    void SetNextUpdate(double n) {fNextUpdate = n;}
    double GetNextUpdate() const {return fNextUpdate;}

    /// Set (get) the acceptance rigidity.  This is the rough exponential
    /// constant needed to relax the acceptance to the target acceptance.  It
    /// is provided as a factor of the acceptance window.  When the rigidity
    /// is more than 100, the step size is fixed and will not be adapted based
    /// on the acceptance rate.  The value can be as small as 0.5, but typical
    /// values should be around 2.0 or greater.  If the acceptance rigidity is
    /// set to a negative value, the step size will be fixed.
    void SetAcceptanceRigidity(double r) {fAcceptanceRigidity = r;}
    double GetAcceptanceRigidity() const {return fAcceptanceRigidity;}

    /// The proposal steps are chosen based on an estimate of the covariance
    /// of the posterior.  This method forces the covariance of the proposal
    /// to be updated.  It's automatically called during the run, so (while it
    /// can be) it probably doesn't need to be called in user code.
    void UpdateProposal(bool fromReset = false) {
        if (fromReset) MCMC_DEBUG(1) << "Updating after a reset" << std::endl;
        MCMC_DEBUG(1) << "Update after "
                      << fSuccesses << "/" << fTrials << " successes"
                      << " (Accepting: " << fAcceptance
                      << " w/ width: " << fSigma << ")"
                      << std::endl;

        MCMC_DEBUG(1) << " Covariance estimated with window of"
                      << " " << int(fCovarianceTrials)
                      << std::endl;
        if (MCMC_DEBUG_LEVEL>1 && fLastPoint.size() < 6) {
            fCurrentCov.Print();
        }

        double currentTrace = GetCovarianceTrace();

        MCMC_DEBUG(1) << " Covariance Trace: " << currentTrace
                      << std::endl;
        MCMC_DEBUG(2) << "        = ";
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            MCMC_DEBUG(2) << fCurrentCov(i,i);
            if (i<fLastPoint.size()-1) MCMC_DEBUG(2) << " + ";
            if (i%6 == 5) MCMC_DEBUG(2) << std::endl << "           ";
        }
        MCMC_DEBUG(2) << std::endl;

        // Adjust the step size for the new covariance.
        MCMC_DEBUG(1) << " Update Sigma " << fSigma << " T:" << fSigmaTrace;
        fSigma = fSigma*std::sqrt(fSigmaTrace/currentTrace);
        fSigmaTrace = currentTrace;
        MCMC_DEBUG(1) << " -> " << fSigma << " T:" << fSigmaTrace
                      << std::endl;

        // This could be algebraically simplified, but the calculation becomes
        // less numerically stable with large numbers of successes.  This
        // implementation remains stable as "up" becomes very large.
        double maxUp = fLastPoint.size()*fLastPoint.size();
        double up = 0.5*fSuccesses;
        fNextUpdate = fAcceptanceWindow + maxUp - maxUp/(up+1.0);

        // The proposal is being updated, so deweight the current covariance
        // so that the new information weighs more.
        if (fCovarianceDeweight > 0.0) {
            if (fCovarianceDeweight > 1.0) fCovarianceDeweight = 1.0;
            MCMC_DEBUG(1) << "Deweight current covariance by "
                          << fCovarianceDeweight
                          << std::endl;
            double w = 1.0 - fCovarianceDeweight;
            fCovarianceTrials = std::max(1.0,w*fCovarianceTrials);
            fCovarianceTrials = std::min(fCovarianceTrials,w*fCovarianceWindow);
            fCentralPointTrials = std::max(1.0,w*fCentralPointTrials);
            fCentralPointTrials =
                std::min(fCentralPointTrials,w*fCovarianceWindow);
        }

        MCMC_DEBUG(1) << " Next update: " << fNextUpdate
                      << " Effective trials in previous covariance: "
                      << fCovarianceTrials
                      << "   Window: " << fCovarianceWindow
                      << std::endl;
        MCMC_DEBUG(1) << " Next update: " << fNextUpdate
                      << " Effective trials in central point: "
                      << fCentralPointTrials
                      << std::endl;

        // The proposal is being updated, so deweight the current acceptance
        // so that the new information weighs more.
        if (fAcceptanceDeweight > 0.0) {
            if (fAcceptanceDeweight > 1.0) fAcceptanceDeweight = 1.0;
            double w = 1.0 - fAcceptanceDeweight;
            fAcceptanceTrials = std::max(1.0,w*fAcceptanceTrials);
            fAcceptanceTrials = std::min(fAcceptanceTrials,w*fAcceptanceWindow);
        }

        // Save the covariance that is being used to generate the Cholesky
        // decomposition.
        fSaveCovariance.clear();
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            for (std::size_t j=0; j<i+1; ++j) {
                fSaveCovariance.push_back(fCurrentCov(i,j));
            }
        }

        // The minimum allowed variance for the posterior along any axis.
        double minVar = std::numeric_limits<Parameter>::epsilon();

#ifndef MCMC_SKIP_CHOLESKY_DECOMPOSITION
        // Decompose the current covariance and use it for the proposal.  If
        // this successes, then UpdateProposal() is done.
        TDecompChol chol(fCurrentCov);
        if (chol.Decompose()) {
            MCMC_DEBUG(1) << "Correlation matrix was decomposed" << std::endl;
            fDecomposition = chol.GetU();
            if (MCMC_DEBUG_LEVEL>1 && fLastPoint.size() < 6) {
                fDecomposition.Print();
            }
            MCMC_DEBUG(2) << " Decomposition Diagonal: "
                          << std::endl;
            MCMC_DEBUG(2) << "        = ";
            for (std::size_t i=0; i<fLastPoint.size(); ++i) {
                MCMC_DEBUG(2) << fDecomposition(i,i);
                if (i<fLastPoint.size()-1) MCMC_DEBUG(2) << " + ";
                if (i%6 == 5) MCMC_DEBUG(2) << std::endl << "           ";
            }
            MCMC_DEBUG(2) << std::endl;
            return;
        }
        MCMC_DEBUG(1) << "Covariance matrix decomposition failed"
                      << std::endl;

        // The Cholesky decomposition has failed.  That usually means that the
        // current estimate of the covariance has a one or more pairs of
        // variables that are too correlated, the variance of one of the
        // variables has become to small, or there is a numeric (e.g. nan)
        // problem.  This can happen when the chain gets "stuck" at a point.

        // Check for very small variances, or invalid variances.  The minimum
        // scale of the covariance is relative to the prechain estimate of the
        // variance.  This should be picked to make sure that there aren't
        // numeric problems...
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            // Get the prechain estimate of the variance for the variable.
            double expectedVariance = 1.0;
            if (fProposalType[i].type == 0) {
                expectedVariance = 1.0;
                if (fProposalType[i].param1>0) {
                    expectedVariance = fProposalType[i].param1;
                }
            }
            else if (fProposalType[i].type == 1) {
                // A uniform proposal...
                expectedVariance = fProposalType[i].param2;
                expectedVariance -= fProposalType[i].param1;
                expectedVariance = expectedVariance*expectedVariance/12.0;
            }
            else {
                MCMC_ERROR << "Illegal proposal type for dimension " << i
                           << ": Type is " << fProposalType[i].type
                           << std::endl;
                throw std::invalid_argument("Illegal proposal type");
            }
            if (!std::isfinite(fCurrentCov(i,i))) {
                fCurrentCov(i,i) = expectedVariance;
                MCMC_DEBUG(1) << "Variance for dimension " << i
                              << " is not a finite number.  Set to "
                              << fCurrentCov(i,i)
                              << std::endl;
            }
            if (fCurrentCov(i,i) < 0.0) {
                fCurrentCov(i,i) = minVar*expectedVariance;
                MCMC_DEBUG(1) << "Variance for dimension " << i
                              << " is negative.  Set to "
                              << fCurrentCov(i,i)
                              << std::endl;
            }
            if (fCurrentCov(i,i) < minVar*expectedVariance) {
                MCMC_DEBUG(1) << "Variance for dimension " << i
                              << " has been increased from " << fCurrentCov(i,i)
                              << " to " << minVar*expectedVariance
                              << std::endl;
                fCurrentCov(i,i) = minVar*expectedVariance;
            }
            if (fCurrentCov(i,i) < minVar) {
                MCMC_DEBUG(1) << "Variance for dimension " << i
                              << " has underflow. Set from " << fCurrentCov(i,i)
                              << " to " << minVar
                              << std::endl;
                fCurrentCov(i,i) = minVar;
            }
        }

        // Check for very large correlations and other numeric problems with
        // the correlations.
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            for (std::size_t j=i+1; j<fLastPoint.size(); ++j) {
                double correlation = fCurrentCov(i,j);
                correlation /= std::sqrt(fCurrentCov(i,i));
                correlation /= std::sqrt(fCurrentCov(j,j));
                // non finite correlations are zero.
                if (!std::isfinite(correlation)) {
                    MCMC_DEBUG(1) << "Correlation between dimension " << i
                                  << " and " << j
                                  << " is not finite.  Set " << correlation
                                  << " to " << 0.0
                                  << std::endl;
                    correlation = 0.0;
                }
                // Only worry about "large" correlations.
                if (std::abs(correlation) > fMaxCorrelation) {
                    // Oops, the correlation is too large, so reduce it..
                    MCMC_DEBUG(1) << "Correlation between dimension " << i
                                  << " and " << j
                                  << " has been reduced from " << correlation;
                    if (correlation > 0.0) correlation = fMaxCorrelation;
                    else correlation = - fMaxCorrelation;
                    MCMC_DEBUG(1) << " to " << correlation
                                  << std::endl;
                }
                fCurrentCov(i,j) = correlation;
                fCurrentCov(i,j) *= std::sqrt(fCurrentCov(i,i));
                fCurrentCov(i,j) *= std::sqrt(fCurrentCov(j,j));
                fCurrentCov(j,i) = fCurrentCov(i,j);
            }
        }

        // Make another attempt at finding the Cholesky decomposition.
        TDecompChol chol2(fCurrentCov);
        if (chol2.Decompose()) {
            MCMC_DEBUG(1) << "Correlation matrix was decomposed"
                          << " after conditioning"
                          << std::endl;
            fDecomposition = chol2.GetU();
            if (MCMC_DEBUG_LEVEL>1 && fLastPoint.size() < 6) {
                fDecomposition.Print();
            }
            MCMC_DEBUG(2) << " Decomposition Diagonal: "
                          << std::endl;
            MCMC_DEBUG(2) << "        = ";
            for (std::size_t i=0; i<fLastPoint.size(); ++i) {
                MCMC_DEBUG(2) << fDecomposition(i,i);
                if (i<fLastPoint.size()-1) MCMC_DEBUG(2) << " + ";
                if (i%6 == 5) MCMC_DEBUG(2) << std::endl << "           ";
            }
            MCMC_DEBUG(2) << std::endl;
            return;
        }

        MCMC_DEBUG(1) << "Covariance decomposition failed after conditioning"
                      << std::endl;
#endif

#ifndef MCMC_SKIP_EIGENVALUE_DECOMPOSITION
        /// Decompose the covariance using Eigenvalue decomposition.  This is
        /// more robust than Cholesky decomposition because it doesn't fail
        /// when the matrix is not positive definite.  When an eigenvalue is
        /// negative, the covariance was not positive definite.  This is
        /// managed by making sure that the decomposition is adjusted so the
        /// eigenvalues are never less than zero.
        TMatrixDSym conditioned(fLastPoint.size());
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            for (std::size_t j=i; j<fLastPoint.size(); ++j) {
                conditioned(j,i) = conditioned(i,j) = fCurrentCov(i,j);
            }
        }

        // The fast option didn't work, so use eigen value decomposition.
        // This will rotate to the best basis, but is slow.
        TMatrixDSymEigen makeEigenVectors(conditioned);
        TMatrixD eigenVectors(makeEigenVectors.GetEigenVectors());
        TVectorD eigenValues(makeEigenVectors.GetEigenValues());

        // Make the eigenValues positive.  Negative values are OK by symmetry,
        // but make positive so the square-root can be taken.  Also check that
        // the eigenvalues don't get too small (protects against losing a
        // dimension).
        double eigenSum = 0.0;
        MCMC_DEBUG(1) << "Eigenvalues: ";
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            MCMC_DEBUG(1) << eigenValues(i);
            if (i<fLastPoint.size()-1) MCMC_DEBUG(1) << ", ";
            if ((i%6) == 5) MCMC_DEBUG(1) << std::endl << "    ";
            if (eigenValues(i) < 0.0) continue;
            eigenSum += eigenValues(i);
        }
        MCMC_DEBUG(1) << std::endl;
        MCMC_DEBUG(1) << "Sum of eigenvalues: " << eigenSum << std::endl;

        // Determine the minimum variance along any axis.  This is set using
        // the maximum allowed correlation and the magnitude of the first
        // (i.e. maximum) eigenvalue.  The correlation between two
        // eigenvectors is one minus the ratio of the eigenvalues.
        double minAxis = 1.0-fMaxCorrelation;
        if (minAxis < minVar) minAxis = minVar;
        minAxis = minAxis*eigenValues(0);

        // Copy into the decomposition and use the eigenvalue to adjust the
        // magnitude of the eigenvector to be equal to the RMS along this
        // direction.  The decomposion is a transpose of the eigenvector
        // matrix so that the resulting matrix can be used in the same way as
        // the Cholesky decmposition.
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            // This is where negative eigenvalues are managed.  The minAxis
            // value is always greater than zero.
            double rms = std::max(minAxis,eigenValues(i));
            rms = std::sqrt(rms);
            for (std::size_t j=0; j<fLastPoint.size(); ++j) {
                // Notice that the elements are being transposed!
                fDecomposition(i,j) = rms*eigenVectors(j,i);
            }
        }

        // At high debug levels print all of the eigenvalues and eigenvectors.
        if (MCMC_DEBUG_LEVEL>1 && fLastPoint.size() < 6) {
            for (std::size_t i=0; i<fLastPoint.size(); ++i) {
                std::cout << eigenValues(i) << " --";
                for (std::size_t j = 0; j <fLastPoint.size(); ++j) {
                    std::cout << " " << eigenVectors(j,i);
                }
                std::cout << std::endl;
            }
        }

        MCMC_DEBUG(1) << "   Sum of eigen values: " << eigenSum << std::endl;

        // Make sure there were positive eigenvalues.  The only way that isn't
        // true seems to be when the decomposition failed and all the
        // eigenValues are zero.
        if (eigenSum > 1E-6) return;

        MCMC_DEBUG(1) << "Eigenvalue decomposition failed" << std::endl;
#endif

        // We are in serious trouble because the covariance wasn't positive
        // definite causing Cholesky decomposition to failed, and none of the
        // eigen values were positive (mathematically, that can't be true, but
        // can happen due to numeric error build up).  Try reducing the
        // correlations and increasing the variances in steps.  This is a last
        // ditch effort!  If this fails, the covariance matrix will be reset
        // to the initial value.

        // Set the amount to increast the variances by at each trial.
        double step = std::numeric_limits<Parameter>::epsilon();
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            step = std::max(step,fCurrentCov(i,i));
        }
        step *= 1E-4;

        double dec = 1.0;   // The amount to decrease the correlation by.
        double total = 1.0; // The total decrease in the correlation.
        for (int trial = 0; trial<10; ++trial) {
            dec *= 0.84; // about sqrt(sqrt(2)), but value is not very important
            total *= dec;
            MCMC_DEBUG(1) << "Reducing correlations by " << total
                          << " and increasing variances by " << step
                          << std::endl;
            for (std::size_t i=0; i<fLastPoint.size(); ++i) {
                fCurrentCov(i,i) += step;
                for (std::size_t j=i+1; j<fLastPoint.size(); ++j) {
                    fCurrentCov(i,j) = fCurrentCov(j,i) = dec*fCurrentCov(i,j);
                }
            }
            // Try the decomposition again.  This will work if the matrix has
            // become positive definite.
            TDecompChol chol3(fCurrentCov);
            if (chol3.Decompose()) {
                MCMC_DEBUG(1) << "Correlation matrix was decomposed in"
                              << " emergency trial " << trial
                              << std::endl;
                fDecomposition = chol3.GetU();
                if (MCMC_DEBUG_LEVEL>1 && fLastPoint.size() < 6) {
                    fDecomposition.Print();
                }
                MCMC_DEBUG(2) << " Decomposition Diagonal: "
                              << std::endl;
                MCMC_DEBUG(2) << "        = ";
                for (std::size_t i=0; i<fLastPoint.size(); ++i) {
                    MCMC_DEBUG(2) << fDecomposition(i,i);
                    if (i<fLastPoint.size()-1) MCMC_DEBUG(2) << " + ";
                    if (i%6 == 5) MCMC_DEBUG(2) << std::endl << "           ";
                }
                MCMC_DEBUG(2) << std::endl;
                return;
            }
        }

        // If this is attempting to work with the user provided correlation
        // matrix after a reset, then throw.  This will happen at the very
        // beginning of the run if the inputs are invalid and can only be
        // fixed by changing the inputs.
        if (fromReset) {
            throw std::runtime_error(
                "Decomposition of user correlations failed");
        }

        // Something is has gone very wrong, so reset the Proposal.
        ResetProposal();
    }

    /// Forget information about the covariance, and use the last point as the
    /// new central value.  This can be useful after burnin to completely
    /// forget about the path to stocastic equilibrium since the covariance is
    /// reset back to the initial conditions.
    void ResetProposal() {
        MCMC_DEBUG(1) << "Reset the proposal after "
                      << fSuccesses << " successes "
                      << " in " << fTrials << " trials "
                      << std::endl;
        MCMC_DEBUG(1) << " Recent acceptance rate was " << fAcceptance
                      << " with an adjusted width of " << fSigma
                      << std::endl;
        // Reset the success and trials counts.
        fTrials = 0;
        fSuccesses = 0;
        // Take a wild guess at width to get the right acceptance.
        if (fSigma < 0.01*std::sqrt(1.0/fLastPoint.size())) {
            fSigma = std::sqrt(1.0/fLastPoint.size());
        }
        // Setup the spece for the decomposition.
        fDecomposition.ResizeTo(fLastPoint.size(), fLastPoint.size());
        // Set up the initial estimate of the covariance.
        fCurrentCov.ResizeTo(fLastPoint.size(), fLastPoint.size());
        for (std::size_t i = 0; i < fLastPoint.size(); ++i) {
            for (std::size_t j = i; j < fLastPoint.size(); ++j) {
                if (i == j
                    && fProposalType[i].type == 0
                    && fProposalType[i].param1 > 0) {
                    MCMC_DEBUG(2) << "Overriding covariance for dimension "
                                  << i
                                  << " from "
                                  << fCurrentCov(i,i)
                                  << " to "
                                  << fProposalType[i].param1
                                  << std::endl;
                    fCurrentCov(i,i) = fProposalType[i].param1;
                }
                else if (i == j && fProposalType[i].type == 1) {
                    MCMC_DEBUG(2) << "Overriding covariance for "
                                  << i
                                  << " to uniform "
                                  << std::endl;
                    double delta = fProposalType[i].param1;
                    delta -= fProposalType[i].param2;
                    fCurrentCov(i,i) = delta*delta/12.0;
                }
                else if (i == j) {
                    fCurrentCov(i,i) = 1.0;
                }
                else fCurrentCov(i,j) = fCurrentCov(j,i) = 0.0;
            }
        }
        // Apply any correlations between the dimensions.
        for (std::vector<CorrelationRecord>::iterator c = fCorrelations.begin();
             c != fCorrelations.end(); ++c) {
            if (c->dim1 == c->dim2) {
                MCMC_DEBUG(0) << "Correlations must be for different dimensions"
                              << std::endl;
                continue;
            }
            double v1 = fCurrentCov(c->dim1,c->dim1);
            double v2 = fCurrentCov(c->dim2,c->dim2);
            fCurrentCov(c->dim1,c->dim2)
                = fCurrentCov(c->dim2,c->dim1)
                = c->correlation*std::sqrt(v1)*std::sqrt(v2);
        }

        // Save the trace of the initial covariance
        fSigmaTrace = GetCovarianceTrace();

        // Set a default window to average the covariance over.  This
        // basically averages over everything.  It shouldn't be infinite since
        // the way the covariance is calculated starts to run into round-off
        // errors if the is overly large.  If the current window is less than
        // minWindow, then it's not initialized or the user is asking for a
        // silly value, and "we know better".
        int minWindow = 100 + 4*fLastPoint.size();
        if (fCovarianceWindow < minWindow) {
            fCovarianceWindow = fLastPoint.size();
            fCovarianceWindow *= fLastPoint.size();
            fCovarianceWindow *= fLastPoint.size();
            fCovarianceWindow += minWindow;
            double r = std::numeric_limits<Parameter>::epsilon();
            fCovarianceWindow = std::min(fCovarianceWindow,std::sqrt(1.0/r));
        }
        // After reseting, erase the acceptance history.
        if (fTargetAcceptance < 0.0) {
            throw std::runtime_error("Target acceptance not initialized");
        }
        fAcceptance = fTargetAcceptance;
        fAcceptanceTrials = std::min(10.0, 0.5*fAcceptanceWindow);
        // Initialize the central point.
        fCentralPoint.resize(fLastPoint.size());
        std::copy(fLastPoint.begin(), fLastPoint.end(), fCentralPoint.begin());
        // Make sure the central point change is resized and starts at zero
        fCentralPointChange.clear();
        fCentralPointChange.resize(fLastPoint.size(),0.0);
        // Start with a non-zero number of trials to represent the prior
        // information coming from the first guess at the central point.
        fCentralPointTrials = std::max(fCentralPointTrials,1.0);
        // This makes sure everything is set properly.
        UpdateProposal(true);
    }

    /// Restore the proposal state so that an existing chain can be continued.
    /// The tree parameter provides a pointer to a tree that was generated by
    /// a previous run of TSimpleMCMC.  This attaches to the tree, gets the
    /// required values, and then detaches (by setting the branch address to
    /// NULL).
    bool RestoreState(const Vector& current, const double value,
                      TTree* tree) {
        fStateInitialized = true;

        if (fLastPoint.size() < 1) {
            SetDim(current.size());
        }
        else if (fLastPoint.size() != current.size()) {
            // Sanity check! These must be equal.
            MCMC_ERROR << "Mismatch in the dimensionality."
                       << std::endl;
        }
        fLastValue = value;
        std::copy(current.begin(), current.end(), fLastPoint.begin());

        if (!tree) return false;
        tree->SetBranchAddress("AdaptiveTrials",&fSaveTrials);
        tree->SetBranchAddress("AdaptiveSuccesses",&fSaveSuccesses);
        tree->SetBranchAddress("AdaptiveNextUpdate",&fSaveNextUpdate);
        tree->SetBranchAddress("AdaptiveAcceptance",&fSaveAcceptance);
        tree->SetBranchAddress("AdaptiveAcceptanceTrials",
                               &fSaveAcceptanceTrials);
        tree->SetBranchAddress("AdaptiveSigma",&fSaveSigma);
        Vector* addrSaveCentralPoint = &fSaveCentralPoint;
        tree->SetBranchAddress("AdaptiveCentralPoint",&addrSaveCentralPoint);
        tree->SetBranchAddress("AdaptiveCentralPointTrials",
                               &fSaveCentralPointTrials);
        Vector* addrSaveCovariance = &fSaveCovariance;
        tree->SetBranchAddress("AdaptiveCovariance",&addrSaveCovariance);
        tree->SetBranchAddress("AdaptiveCovarianceTrace",&fSaveTrace);
        tree->SetBranchAddress("AdaptiveCovarianceTrials",
                               &fSaveCovarianceTrials);

        // The state should be saved in the last entry in the tree.  If it's
        // not there, then this will look backwards until it finds the state,
        // or runs out of tree.  If the last entry does not contain the state,
        // this will produce lots of warnings!!!!
        int entries = tree->GetEntries();
        int elem = entries;
        int dim = fLastPoint.size();
        std::size_t covSize = dim*(dim+1)/2;
        while (elem > 1) {
            --elem;
            tree->GetEntry(elem);
            if (fSaveCovariance.size() != covSize) {
                MCMC_DEBUG(2) << "Covariance not found at entry " << elem
                              << "/" << entries
                              << std::endl;
                continue;
            }
            break;
        }

        MCMC_DEBUG(1) << "Chain state restored from entry "
                      << elem << "/" << entries
                      << std::endl;

        fTrials = fSaveTrials;
        fSuccesses = fSaveSuccesses;
        fNextUpdate = fSaveNextUpdate;
        fAcceptance = fSaveAcceptance;
        fAcceptanceTrials = fSaveAcceptanceTrials;
        fSigma = fSaveSigma;
        fCentralPoint = fSaveCentralPoint;
        fCentralPointTrials = fSaveCentralPointTrials;

        // The covariance was saved in a vector, so it has to be unpacked.
        // This must match the code in SaveState().
        Vector::const_iterator cov = fSaveCovariance.begin();
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            for (std::size_t j=0; j<i+1; ++j) {
                if (cov == fSaveCovariance.end()) {
                    MCMC_ERROR << "Past the end of the covariance"
                               << std::endl;
                    throw std::logic_error("Past the end of the covariance");
                }
                if (i == j) fCurrentCov(i,i) = *cov;
                else fCurrentCov(i,j) = fCurrentCov(j,i) = *cov;
                ++cov;
            }
        }
        fSigmaTrace = GetCovarianceTrace();
        fCovarianceTrials = fSaveCovarianceTrials;

        // Reset the branch addresses.
        tree->SetBranchAddress("AdaptiveTrials",NULL);
        tree->SetBranchAddress("AdaptiveSuccesses",NULL);
        tree->SetBranchAddress("AdaptiveNextUpdate",NULL);
        tree->SetBranchAddress("AdaptiveAcceptance",NULL);
        tree->SetBranchAddress("AdaptiveAcceptanceTrials",NULL);
        tree->SetBranchAddress("AdaptiveSigma",NULL);
        tree->SetBranchAddress("AdaptiveCentralPoint",NULL);
        tree->SetBranchAddress("AdaptiveCentralPointTrials",NULL);
        tree->SetBranchAddress("AdaptiveCovariance",NULL);
        tree->SetBranchAddress("AdaptiveCovarianceTrace",NULL);
        tree->SetBranchAddress("AdaptiveCovarianceTrials",NULL);

        MCMC_DEBUG(1) << "Restored state"
                      << " T: " << fTrials
                      << " S: " << fSuccesses
                      << " A: " << fAcceptance
                      << " Sig: " << fSigma
                      << " Trace: " << GetCovarianceTrace()
                      << std::endl;

        // Now update the proposal.
        UpdateProposal();

        return true;
    }

    /// This attaches any branches needed to save the state to the output
    /// tree.
    bool AttachState(TTree *tree) {
        if (!tree) return false;
        tree->Branch("AdaptiveTrials",&fSaveTrials);
        tree->Branch("AdaptiveSuccesses",&fSaveSuccesses);
        tree->Branch("AdaptiveNextUpdate",&fSaveNextUpdate);
        tree->Branch("AdaptiveAcceptance",&fSaveAcceptance);
        tree->Branch("AdaptiveAcceptanceTrials",&fSaveAcceptanceTrials);
        tree->Branch("AdaptiveSigma",&fSaveSigma);
        tree->Branch("AdaptiveCentralPoint",&fSaveCentralPoint);
        tree->Branch("AdaptiveCentralPointTrials",&fSaveCentralPointTrials);
        tree->Branch("AdaptiveCovariance",&fSaveCovariance);
        tree->Branch("AdaptiveCovarianceTrace",&fSaveTrace);
        tree->Branch("AdaptiveCovarianceTrials",&fSaveCovarianceTrials);
        return true;
    }

    /// Fill the output tree with the current state.
    bool SaveState(bool fullSave=false) {
        fSaveTrials = fTrials;
        fSaveSuccesses = fSuccesses;
        fSaveNextUpdate = fNextUpdate;
        fSaveAcceptance = fAcceptance;
        fSaveAcceptanceTrials = fAcceptanceTrials;
        fSaveSigma = fSigma;
        fSaveTrace = GetCovarianceTrace();
        fSaveCentralPointTrials = fCentralPointTrials;
        fSaveCovarianceTrials = fCovarianceTrials;
        fSaveCentralPoint.clear();
        fSaveCovariance.clear();
        if (!fullSave) return false;
        fSaveCentralPoint = fCentralPoint;
        for (std::size_t i=0; i<fLastPoint.size(); ++i) {
            for (std::size_t j=0; j<i+1; ++j) {
                fSaveCovariance.push_back(fCurrentCov(i,j));
            }
        }

        return true;
    }

    /// Notification that the last saved state has been written to the output.
    /// This can be used by the proposal class to zero out the tree branches.
    bool StateSaved() {
#define MCMC_ZERO_ADAPTIVE_STEP_SAVED_STATE
#ifdef MCMC_ZERO_ADAPTIVE_STEP_SAVED_STATE
        /// The state of the adaptive step is not usually updated between
        /// steps, so the same values can be written to the output tree many
        /// times.  The ROOT compression usually does a pretty good job of
        /// recovering the space, but this might sometimes reduce the output
        /// size.
        fSaveTrials = 0;
        fSaveSuccesses = 0;
        fSaveNextUpdate = 0;
        fSaveAcceptance = 0;
        fSaveAcceptanceTrials = 0;
        fSaveSigma = 0;
        fSaveCentralPointTrials = 0;
        fSaveCovarianceTrials = 0;
        fSaveCentralPoint.clear();
        fSaveCovariance.clear();
#endif
        return false;
    }

    // Return to a default state.
    void InitializeState(const Vector& current, const double value) {
        if (fStateInitialized) return;
        fStateInitialized = true;
        if (fLastPoint.size() < 1) {
            SetDim(current.size());
        }
        else if (fLastPoint.size() != current.size()) {
            // Sanity check! These must be equal.
            MCMC_ERROR << "Mismatch in the dimensionality."
                       << std::endl;
        }
        fLastValue = value;
        std::copy(current.begin(), current.end(), fLastPoint.begin());
        // Set a default window to average the acceptance over.
        if (fAcceptanceWindow < 0) {
            fAcceptanceWindow = std::pow(1.0*fLastPoint.size(),1.5) + 1000;
        }
        // The steps until the next update.
        fNextUpdate = fAcceptanceWindow;
        // Make sure we have a reasonable target acceptance
        if (fTargetAcceptance < 1E-4) {
            // Set a default value for the target acceptance rate.  For some
            // reason, the magic value in the literature is 44% for 1D and
            // 23.4% For more than several (~5) dimensions.  The default value
            // is set for high dimensions.  In fact, 23.4% is really suppose
            // to be an upper bound on the optimal acceptance rate at n-D.
            // See https://doi.org/10.1016/j.spa.2007.12.005 "Optimal
            // acceptance rates for Metropolis algorithms: Moving beyond
            // 0.234" (Stochastic Processes and their Applications, Volume
            // 118, Issue 12, 2198-2222, 2008
            if (fLastPoint.size() > 4) fTargetAcceptance = 0.234;
            else fTargetAcceptance = 0.44;
        }
        // Reset the proposal covariance as part of the initialization.
        ResetProposal();
    }

private:

    /// This updates the current state.  The new proposals are adjusted based
    /// on the past history of success (or failure).  This helps make the
    /// chain more efficient at exploring the posterior.
    void UpdateState(const Vector& current, const double value) {
        InitializeState(current,value);
        ++fTrials;

        // Make a quick check to see if the point moved.  This gets it right
        // more than 99.9% of the time (i.e. it's Good Enough (tm)).
        bool accepted = false;
        if (value != fLastValue || current[0] != fLastPoint[0]) accepted = true;

        // Track the total number of successes.
        if (accepted) ++fSuccesses;

        // Update the acceptance over the last "fAcceptanceWindow" trials.
        fAcceptance *= fAcceptanceTrials;
        if (accepted) fAcceptance = fAcceptance+1.0;
        fAcceptance /= fAcceptanceTrials + 1.0;
        fAcceptanceTrials = std::min(fAcceptanceWindow,fAcceptanceTrials+1.0);

        // Update the rigidity based on how close the acceptance is to the
        // target acceptance.  If the acceptance is close to the target
        // acceptance, then increase the rigidity by a little.  If the
        // acceptance is far from the target, then decrease the rigidity.
        // The acceptance sigma is the fractional standard deviation of a
        // binomial distribution within fAcceptanceWindow samples.
        if (fAcceptanceRigidity < 500.0 && fAcceptanceRigidity > 0.0) {
            double acceptanceSigma = fTargetAcceptance*(1.0-fTargetAcceptance);
            acceptanceSigma = std::sqrt(acceptanceSigma/fAcceptanceWindow);
            if (std::abs(fAcceptance-fTargetAcceptance)<acceptanceSigma) {
                // Within one standard deviation of the target acceptance, so
                // increase the rigidity.
                fAcceptanceRigidity
                    += 0.5*fAcceptanceRigidity/fAcceptanceWindow;
                fAcceptanceRigidity = std::min(200.0,fAcceptanceRigidity);
            }
            if (std::abs(fAcceptance-fTargetAcceptance)>4.0*acceptanceSigma) {
                // More than four standard deviations from the target
                // acceptance, so decrease the rigidity.
                fAcceptanceRigidity
                    -= 1.618*0.5*fAcceptanceRigidity/fAcceptanceWindow;
                fAcceptanceRigidity = std::max(2.0,fAcceptanceRigidity);
            }
        }

        // Update the scale for the proposal step.  If the acceptance is to
        // small, then the scale will be reduced.  If the acceptance is to
        // large, then the scale will be increased.  If the ratio of measured
        // acceptance to the target acceptance is one, then the value of
        // fSigma will not change.  A larger rigidity means that the fSigma
        // value will change more slowly, and if the rigidity gets too large
        // the fSigma value is fixed.
        if (fAcceptanceRigidity > 0 && fAcceptanceRigidity < 100.0) {
            fSigma *=
                std::pow(fAcceptance/fTargetAcceptance,
                         std::min(1.0/500.0,
                                  1.0/(fAcceptanceRigidity*fAcceptanceWindow)));
        }

        // Update the estimate of the central value.  This is simply a running
        // average of the position of the points in the posterior.
        for (std::size_t i=0; i< current.size(); ++i) {
            fCentralPointChange[i] = fCentralPoint[i];
            fCentralPoint[i] *= fCentralPointTrials;
            fCentralPoint[i] += current[i];
            fCentralPoint[i] /= fCentralPointTrials + 1;
            fCentralPointChange[i] = fCentralPoint[i] - fCentralPointChange[i];
        }
        fCentralPointTrials = std::min(fCovarianceWindow,
                                       fCentralPointTrials+1.0);

        // Update the estimate of the covariance.  This is a running
        // calculation of the covariance.  This does a *VERY* approximate
        // calculation of the covariance relative to the current idea of the
        // central point.  It's OK to use as a guess of the next step, but
        // that's about all.
        for (std::size_t i=0; i<current.size(); ++i) {
            for (std::size_t j=0; j<i+1; ++j) {
                double v = fCurrentCov(i,j);
                // Correct for the center having moved!
                v += fCentralPointChange[i]*fCentralPoint[j];
                v += fCentralPointChange[j]*fCentralPoint[i];
                v -= fCentralPointChange[i]*fCentralPointChange[j];
                // Calculate the "covariance" of the new point.
                double r = (current[i]-fCentralPoint[i])
                    *(current[j]-fCentralPoint[j]);
                // Update the running average of the covariance.
                v *= fCovarianceTrials;
                v += r;
                v /= fCovarianceTrials + 1.0;
                // Make sure the covariance stays symmetric
                if (i == j) fCurrentCov(i,j) = v;
                else fCurrentCov(i,j) = fCurrentCov(j,i) = v;
            }
        }
        fCovarianceTrials = std::min(fCovarianceWindow,
                                     fCovarianceTrials+1.0);

        // Periodically change how the proposal is updated based on the
        // current estimate of the covariance.
        if (accepted && (--fNextUpdate)<1) {
            UpdateProposal();
        }

        // Save the last value and point.
        fLastValue = value;
        std::copy(current.begin(), current.end(), fLastPoint.begin());
    }

    // The previous current point.  This is used to (among other things) keep
    // track of when the state has changed.
    Vector fLastPoint;

    // The previous log Likelihood.
    double fLastValue;

    // The current (running) estimate for the central value of each parameter.
    Vector fCentralPoint;
    Vector fSaveCentralPoint;

    // The change from the last position of the central point.
    Vector fCentralPointChange;

    // The trials being used for the current estimated central value.  This
    // will be a value between one and fCovarianceWindow.
    double fCentralPointTrials;
    double fSaveCentralPointTrials;

    // The current (running) estimate of the covariance.
    TMatrixD fCurrentCov;

    // A vector to save the value of the current covariance.  The order is
    //    fSaveCovariance.clear();
    //    for (int i=0; i<dim; ++i) {
    //         for (int j=0; j<i+1; ++j) {
    //            fSaveCovariance.push_back(fCurrentCov(i,j));
    //         }
    //    }
    Vector fSaveCovariance;

    // The trials being used for the current estimated covariance.  This
    // will be a value between one and fCovarianceWindow.
    double fCovarianceTrials;
    double fSaveCovarianceTrials;

    // The amount the old covariance should be deweighted after an update.
    double fCovarianceDeweight;

    // The window to average the covariance and central value over.  The
    // covariance window should be several times the autocorrelation period.
    // It should usually be very large so that all to points used to probe the
    // posterior are used.  Setting it to a smaller value means that the local
    // covariance of the posterior will be used.  This might be important if
    // the posterior is extremely non-Gaussian (e.g. it's a "banana
    // posterior").
    double fCovarianceWindow;

    // The current decomposition of the covariance.  This is not updated
    // everytime the fCurrentCov estimate changes.  A Cholesky decomposition,
    // but this only works when the covariance is positive definite.  It might
    // not not be numerically positive definite due to error accumulation.
    // Eigenvalue decomposition can be used, but it is very slow.
    TMatrixD fDecomposition;

    // Record the type of proposal to use for each dimension
    struct ProposalType {
        ProposalType(): type(0), param1(0), param2(0) {}
        int type;      // 0 for Gaussian, 1 for Uniform.
        double param1; // Sigma for Gaussian, Minimum for Uniform
        double param2; // Not used for Gaussian, Maximum for Uniform
    };

    // The type of distribution to draw the proposal for a dimension from.
    std::vector<ProposalType> fProposalType;

    // A record to record the correlations.
    struct CorrelationRecord {
        CorrelationRecord(int d1,int d2, double c)
            : dim1(d1), dim2(d2), correlation(c) {}
        int dim1;
        int dim2;
        double correlation; // The correlation coefficient between d1, and d2.
    };

    // The correlations that have been set.
    std::vector<CorrelationRecord> fCorrelations;

    // The maximum allowed correlation.
    double fMaxCorrelation;

    // The number of times a step has been proposed since the last reset.
    int fTrials;
    int fSaveTrials;

    // The total number of successes since the last reset
    int fSuccesses;
    int fSaveSuccesses;

    // A counter until the next update gets done.
    int fNextUpdate;
    int fSaveNextUpdate;

    // The recent acceptance rate.
    double fAcceptance;
    double fSaveAcceptance;

    // The trials being used for the current estimated acceptance rate.  This
    // will be a value between one and fAcceptanceWindow.
    double fAcceptanceTrials;
    double fSaveAcceptanceTrials;

    // The amount the old acceptance should be deweighted after an update.
    double fAcceptanceDeweight;

    // The window to average the acceptance over.
    double fAcceptanceWindow;

    // The rigidity of the acceptance relaxation
    double fAcceptanceRigidity;

    // The target acceptance rate for the chain.
    double fTargetAcceptance;

    // The current width of the proposal.
    double fSigma;
    double fSaveSigma;

    // The trace of the covariance matrix this fSigma is associated with.
    // This is set with the decomposition is done.
    double fSigmaTrace;

    // Save the current trace for convergence checks
    double fSaveTrace;

    // Keep track of whether we've actually been called.
    bool fStateInitialized;

    // A dimension to "scan" this generates a cloud of points with all other
    // parameters fixed and the "scan" parameter fluctuated around the central
    // point.  If the value is outside the range of parameter indices, it's
    // skipped (it must be between 0 and dim to scan).
    int fScanDimension;
};

// MIT License

// Copyright (c) 2017-2022 Clark McGrew

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
#endif

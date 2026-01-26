////////////////////////////////////////////////////////////////////////////
// A example root macro to make plots of the MCMC results.  To run this code
//
// root file1.root [file2.root file3.root ...] gundamPlotMCMC.C
//
// This creates many (most?) of the plots that you need to diagnose an MCMC
// chain, but that are not specific to a specific analysis.  The main plots
// created are: The posterior distributions for each parameter, the
// auto-correlation for each parameter, the trace for each prior, and some
// summary plots.  All of the plots are weighted by the effective sample size,
// and have errors showing the credible band for the true value around the
// estimate.
//
// This will create png files for all of the plots in the current directory.
//
// See the top parto of the gundamPlotMCMC() function for some control
// variables.  They can usually be left at the default values.
//

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <memory>

#include <TFile.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TTree.h>
#include <TChain.h>
#include <TROOT.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TStyle.h>

TCanvas* gCanvas = NULL;

///////////////////////////////////////////////////////////////////////////////
// Copy parameter information in vectors, easier to use later.  This is read
// from the FitterEngine/fit/parameterSets tree.  These parameters are in the
// order that the parameters are used by the fitter.
///////////////////////////////////////////////////////////////////////////////

/// The names of the parameter sets (as defined by the GUNDAM config files).
std::vector<std::string> gSetName;

/// The number of parameters in the set.
std::vector<int> gSetCount;

/// The offset of the first parameter in the ste.  This is an index into the "gPar<blah>" vectors below.
std::vector<int> gSetOffset;

/// The name of each parameter (set in the GUNDAM configuration).
std::vector<std::string> gParName; // Copy of the parameterName branch

/// The name of the set that contains this parameter
std::vector<std::string> gParSetName;

/// The prior value for each parameter (set in the GUNDAM configuration).
std::vector<double> gParPrior;     // Copy of the parameterPrior branch

/// The prior SIGMA for each parameter (set in the GUNDAM configuration).
std::vector<double> gParSigma;     // Copy of the parameterSigma branch

/// The index of the parameter in the parameter set (not needed here, but kept
/// anyway).
std::vector<int> gParIndex;        // Copy of the parameterIndex branch

/// Flag for if the parameter was enabled during the fit.
std::vector<bool> gParEnabled;     // Copy of the parameterEnabled branch

/// Flag for if the parameter value was fixed during the fit.
std::vector<bool> gParFixed;       // Copy of the parameterFixed branch

/// The minimum analytic value for the parameter.  This might be unphysical
std::vector<double> gParMinimum;  // Copy of the parameterMin branch

/// The maximum analytic value for the parameter.  This might be unphysical.
std::vector<double> gParMaximum;  // Copy of the parameterMax branch

/// The name of the samples
std::vector<std::string> gParSampleName;

/// The offset of the sample histogram in gParSampleData;
std::vector<int> gParSampleOffsets;

/// The data histograms
std::vector<double> gParSampleData;

///////////////////////////////////////////////////////////////////////////
// Save the currently drawn histogram, in png and .C formats. The plot
// will be placed in the appropriate water in or out directory.
void PrintDrawnCanvas(const std::string& basename, int formats=0){
    std::string plotName = basename;
    std::replace(plotName.begin(),plotName.end(),'#','_');
    std::replace(plotName.begin(),plotName.end(),'(','_');
    std::replace(plotName.begin(),plotName.end(),')','_');
    std::replace(plotName.begin(),plotName.end(),' ','_');

    std::string pngName = plotName;
    pngName += ".png";
    std::string pdfName = plotName;
    pdfName += ".pdf";
    std::string cName = plotName;
    cName += ".C";

    gCanvas->Print(pngName.c_str());
    // if (formats & 1) gCanvas->Print(pdfName.c_str());
    // if (formats & 2) gCanvas->Print(cName.c_str());
}

/// Read the FitterEngine/fit/parameterSets tree and save the information into
/// memory.
void readParameterSets() {

    if (!gFile) return;

    // Parameter naming information
    TTree *parSets = (TTree*)gFile->Get("FitterEngine/fit/parameterSets");
    std::vector<std::string> *parSetNames = 0;
    std::vector<int> *parSetOffsets = 0;
    std::vector<int> *parSetCounts = 0;
    std::vector<int> *parIndex = 0;
    std::vector<bool> *parFixed = 0;
    std::vector<bool> *parEnabled = 0;
    std::vector<std::string> *parName = 0;
    std::vector<double> *parPrior = 0;
    std::vector<double> *parSigma = 0;
    std::vector<double> *parMin = 0;
    std::vector<double> *parMax = 0;

    if (not parSets) {
        std::cout << "FitterEngine/fit/parameterSets not found in "
                  << gFile->GetName() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    parSets->SetBranchAddress("parameterSetNames", &parSetNames);
    parSets->SetBranchAddress("parameterSetOffsets", &parSetOffsets);
    parSets->SetBranchAddress("parameterSetCounts", &parSetCounts);
    parSets->SetBranchAddress("parameterIndex", &parIndex);
    parSets->SetBranchAddress("parameterFixed", &parFixed);
    parSets->SetBranchAddress("parameterEnabled", &parEnabled);
    parSets->SetBranchAddress("parameterName", &parName);
    parSets->SetBranchAddress("parameterPrior", &parPrior);
    parSets->SetBranchAddress("parameterSigma", &parSigma);
    parSets->SetBranchAddress("parameterMin", &parMin);
    parSets->SetBranchAddress("parameterMax", &parMax);

    std::vector<std::string> *parSampleName = 0;
    std::vector<int> *parSampleOffsets = 0;
    std::vector<double> *parSampleData = 0;
    parSets->SetBranchAddress("parameterSampleName", &parSampleName);
    parSets->SetBranchAddress("parameterSampleOffsets", &parSampleOffsets);
    parSets->SetBranchAddress("parameterSampleData", &parSampleData);

    // Number of parameter sets
    parSets->GetEntry(0);
    parSets->Print();

    int nSet = parSetNames->size();
    // Save number of parameters in each set
    for (int iSet = 0; iSet < nSet; iSet++) {
        gSetName.push_back(parSetNames->at(iSet));
        gSetCount.push_back(parSetCounts->at(iSet));
        gSetOffset.push_back(parSetOffsets->at(iSet));
        std::cout << "Set " << iSet << " " << gSetName.back()
                  << " O: " << gSetOffset.back()
                  << " N: " << gSetCount.back()
                  << std::endl;
    }

    // Fill a back map between the parameter numbers and the parameter sets
    for (int iSet = 0; iSet < nSet; iSet++) {
        for (int j = 0; j < gSetCount[iSet]; ++j) {
            gParSetName.push_back(gSetName[iSet]);
        }
    }

    for (std::size_t iPar = 0; iPar < parName->size(); ++iPar){
        gParName.push_back(parName->at(iPar));
        gParIndex.push_back(parIndex->at(iPar));
        gParPrior.push_back(parPrior->at(iPar));
        gParSigma.push_back(parSigma->at(iPar));
        gParFixed.push_back(parFixed->at(iPar));
        gParEnabled.push_back(parEnabled->at(iPar));
        gParMinimum.push_back(parMin->at(iPar));
        gParMaximum.push_back(parMax->at(iPar));
        std::cout << iPar
                  << " N: " << gParName.back()
                  << " I: " << gParIndex.back()
                  << " p: " << gParPrior.back()
                  << " s: " << gParSigma.back()
                  << " r: " << gParMinimum.back()
                  << " -> " << gParMaximum.back()
                  << std::endl;
    }

    // Save the data histogram information
    gParSampleName.resize(parSampleName->size());
    std::copy(parSampleName->begin(), parSampleName->end(),
              gParSampleName.begin());

    gParSampleOffsets.resize(parSampleOffsets->size());
    std::copy(parSampleOffsets->begin(), parSampleOffsets->end(),
              gParSampleOffsets.begin());

    gParSampleData.resize(parSampleData->size());
    std::copy(parSampleData->begin(), parSampleData->end(),
              gParSampleData.begin());

}

TH1D* gPosteriorAutocorrelationAvg = nullptr;
TH2D* gPosteriorAutocorrelationHist = nullptr;
std::vector<TH1D*> gPosteriorAutocorrelation;

/////////////////////////////////////////////////////////////////////////
// Do a brute force calculation of the auto-correlations.  This is fast enough
// to use on several 10M's events.  There are faster ways, but this has the
// advantage that it keeps more numeric stability.  There are even more tricks
// that could be used for numeric stability, but it starts to get much more
// complicated (i.e. there are getter ways to accumulate the sums).
void fillAutocorrelations(TChain* mcmcTree, int first,
                          int maxAuto, int autoBins) {
    // std::vector<double> *points =0;
    std::vector<float> *points =0;
    std::vector<double> *accepted =0;
    mcmcTree->SetBranchAddress("Points",&points);
    mcmcTree->GetEntry(0);
    int nPostEntries = mcmcTree->GetEntries();

    if (points->size() != gParName.size()) {
        std::cout << "Problem with input data"
                  << " points " << points->size()
                  << " names " << gParName.size()
                  << std::endl;
        throw;
    }

    int printCounter = 0;

    std::vector<std::vector<double>> ring;
    std::vector<double> sumN;
    std::vector<double> sumY;
    std::vector<double> varY;
    std::vector<double> sumYY;
    std::vector<std::vector<double>> sumIK;
    std::vector<std::vector<double>> sumNIK;

    // Control the number of bins in the autocorrelation plots.
    int autoSampling = std::min(1,std::max(1,maxAuto/autoBins));
    int ringDepth = autoBins*autoSampling;
    int ringStride = maxAuto/ringDepth;
    if (ringStride < 1) ringStride = 1;

    // Space for sums to calculate autocorrelations
    ring.resize(ringDepth);
    for (std::size_t i = 0; i < ring.size(); ++i) {
        ring[i] = std::vector<double>(points->size());
    }
    sumN.resize(points->size());
    sumY.resize(points->size());
    varY.resize(points->size());
    sumIK.resize(points->size());
    sumNIK.resize(points->size());
    for (std::size_t i = 0; i < points->size(); ++i) {
        sumIK[i] = std::vector<double>(ring.size());
        sumNIK[i] = std::vector<double>(ring.size());
    }
    gPosteriorAutocorrelation.resize(sumY.size());

    for (int i = first; i < nPostEntries; i += ringStride) {
        mcmcTree->GetEntry(i);
        if ((printCounter++%1000) == 0) {
            std::cout << "Calculate averages with entry "<< i <<std::endl;
        }
        for (std::size_t iAcc = 0; iAcc < points->size(); ++iAcc) {
            double content = points->at(iAcc);
            sumN[iAcc] += 1;
            sumY[iAcc] += content;
        }
    }
    for (std::size_t iAcc = 0; iAcc < points->size(); ++iAcc) {
        sumY[iAcc] /= sumN[iAcc];
    }

    int ringNext = 0;
    int ringLast = 0;
    for (int i = first; i < nPostEntries; i += ringStride) {
        mcmcTree->GetEntry(i);
        if ((printCounter++%1000) == 0) {
            std::cout << "Fill autocorrelation with entry "<< i
                      << " " << ring.size()
                      << " " << maxAuto
                      << " " << ringStride
                      <<std::endl;
        }
        for (std::size_t iAcc = 0; iAcc < points->size(); ++iAcc) {
            double content = points->at(iAcc);
            double diff = content-sumY[iAcc];
            varY[iAcc] += diff*diff;
            ring[ringNext][iAcc] = content;
        }
        ringNext = (ringNext+1) % ring.size();
        if (ringLast == ringNext) ringLast = (ringLast+1) % ring.size();
        for (std::size_t iAcc = 0; iAcc < points->size(); ++iAcc) {
            double content = points->at(iAcc);
            for (std::size_t k = 0; k < ring.size(); ++k) {
                int ringLoc = ringNext-k-1;
                if (ringLoc < 0) ringLoc += ring.size();
                ringLoc = ringLoc%ring.size();
                if (ringLoc == ringLast) break;
                sumIK[iAcc][k]
                    += (content-sumY[iAcc])*(ring[ringLoc][iAcc]-sumY[iAcc]);
                sumNIK[iAcc][k] += 1;
            }
        }
    }

    for (std::size_t iAcc = 0; iAcc < sumY.size(); ++iAcc) {
        varY[iAcc] /= sumN[iAcc];
        for (std::size_t iLag = 0; iLag < ring.size(); ++iLag) {
            if (sumNIK[iAcc][iLag] < 2) sumIK[iAcc][iLag] = 0.0;
            else sumIK[iAcc][iLag] /= sumNIK[iAcc][iLag];
        }
    }

    /////////////////////////////////////////////////////////////////
    // Calculate and fill the autocorrelations based on collected data.
    /////////////////////////////////////////////////////////////////
    for (std::size_t iAcc = 0; iAcc < sumY.size(); ++iAcc) {
        // Skip parameters that will not be in the summary plots.
        if (!std::isfinite(varY[iAcc])) continue;
        if (varY[iAcc] <= 0.0) continue;

        // Name of current parameter
        std::ostringstream histName;
        histName << "Auto " << gParSetName[iAcc] << " " << gParName[iAcc]
                 << " #" << std::setfill('0') << std::setw(3)  << iAcc;
        std::ostringstream histTitle;
        histTitle << "Auto correlation "
                  << gParSetName[iAcc] << " " << gParName[iAcc]
                  << " Par #" << std::setfill('0') << std::setw(3)  << iAcc;

        gPosteriorAutocorrelation[iAcc]
            = new TH1D(histName.str().c_str(),
                       histTitle.str().c_str(),
                       autoBins,0.0,maxAuto);
        gPosteriorAutocorrelation[iAcc]->GetYaxis()->SetRangeUser(-1.5,1.5);

        for (int k = 0; k<autoBins; ++k) {
            int iRing = k*autoSampling;
            if (iRing >= (int) ring.size()) continue;
            if (sumNIK[iAcc][iRing] < 3) continue;
            double a = sumIK[iAcc][iRing]/varY[iAcc];
            // add uncertainty of corrIK and varY in quadrature
            double s = std::sqrt(5.0*(1.0/sumNIK[iAcc][iRing]
                                      + 1.0/sumN[iAcc]));
            if (!std::isfinite(a)) {
                std::cout << "Invalid correlation for lag " << k
                          << " of "
                          << gParSetName[iAcc] << "/" << gParName[iAcc]
                          << std::endl;
                continue;
            }
            // std::cout << iAcc << " k " << k << " " << iRing
            //           << " c " << sumIK[iAcc][iRing]
            //           << " v " << varY[iAcc]
            //           << " a " << a
            //           << " s " << s
            //           << " nRing " << sumNIK[iAcc][iRing]
            //           << " N " << sumN[iAcc]
            //           << std::endl;
            gPosteriorAutocorrelation[iAcc]->SetBinContent(k+1,a);
            gPosteriorAutocorrelation[iAcc]->SetBinError(k+1,s);
        }
    }
}

void gundamPlotMCMC() {
    if (!gFile) {
        std::cout << "No files attached"
                  << std::endl;
        return;
    }

    int printCounter = 0;   // Just increment and limit the amount of output

    readParameterSets();

    //////////////////////////////////////////////////////////////////
    // Attach to the MCMC input files.
    TChain* mcmcTree = new TChain("FitterEngine/fit/MCMC");
    {
        TSeqCollection *inputFiles = gROOT->GetListOfFiles();
        for (auto f = inputFiles->begin(); f != inputFiles->end(); ++f) {
            TFile* file = dynamic_cast<TFile*>(*f);
            std::cout << "ADD FILE: " << file->GetName() << std::endl;
            mcmcTree->Add(file->GetName());
        }
    }


    // Chose what will be filled
    bool fillTraces = true; // Set to fill the traces
    bool plotTraces = false; // Set to fill the traces
    bool plotParams = true; // Set to make the parameter plots
    bool plotAuto = true;  // Set to fill the auto correlations

    // Number of steps to (hopefully) use from the end of the chain.
    int steps = 1E+9;

    ////////////////////////////////////////////////////////////
    // Control how the steps in the chain are used.
    ////////////////////////////////////////////////////////////

    // The number of entries in the tree.  This is saved since "GetEntries()"
    // is fairly slow.
    int nPostEntries = mcmcTree->GetEntries();

    // The maximum fraction of the chain to be used for burning
    double burninFraction = 0.0;

    // The maximum number of steps to be used for burn-in.  Only applies
    // when the plots are suppose to be for more trials than exist in the
    // chain.
    int maxBurnin = 1000000;
    maxBurnin = std::min(maxBurnin,int(burninFraction*nPostEntries));

    // The maximum autocorrelation lag to calculate.
    int maxAuto = std::min(50000,steps/10);

    // The number of entries to be added to the parameter distributions
    int histogramEntries = 1E+9;

    // The first step to be used in the distributions (calculated).  This
    // adjusts steps so that it doesn't go off the end of the chain.
    int first = std::max(maxBurnin, nPostEntries-steps);
    steps = nPostEntries - first;

    // The stride through the chain when building the plots.
    int stride = steps/histogramEntries;
    if (stride < 1) stride = 1;

    // Open the output file for plots.
    std::unique_ptr<TFile> outputFile(new TFile("GundamPlotMCMC.root","NEW"));
    if (outputFile == nullptr or not outputFile->IsOpen()) {
        std::cout << "Could not open the output file"
                  << std::endl;
        return;
    }
    outputFile->mkdir("Summary");
    outputFile->mkdir("Parameters");
    outputFile->mkdir("Autocorrelations");
    outputFile->mkdir("Traces");

    /// Count the total number of parameters that are defined in the sets.
    int nTotal = 0;
    for (std::size_t iSet = 0; iSet < gSetName.size(); ++iSet) {
        nTotal += gSetCount[iSet];
    }

    if (nTotal != gParName.size()) {
        std::cout << "Bug in the number of parameters" << std::endl;
        return;
    }

    // Find the average, rms, and range for each of the posterior parameters.
    vector<double> posteriorAverage(nTotal);
    vector<double> posteriorRms(nTotal);
    vector<std::pair<double,double>> posteriorRange(nTotal);
    do {
        // std::vector<double> *points =0;
        std::vector<float> *points =0;
        std::vector<double> *accepted =0;
        mcmcTree->SetBranchAddress("Points",&points);
        mcmcTree->GetEntry(0);

        double norm = 0.0;
        for (int i = first; i < nPostEntries; i += stride) {
            mcmcTree->GetEntry(i);
            norm += 1.0;
            if ((printCounter++%10000) == 0) {
                std::cout << "Average parameters at entry " << i << std::endl;
            }
            for (std::size_t iPar = 0; iPar < points->size(); ++iPar) {
                double content = points->at(iPar);
                if (i == first) {
                    posteriorRange[iPar].first = content;
                    posteriorRange[iPar].second = content;
                }
                posteriorAverage[iPar] += content;
                posteriorRms[iPar] += content*content;
                posteriorRange[iPar].first
                    = std::min(content,posteriorRange[iPar].first);
                posteriorRange[iPar].second
                    = std::max(content,posteriorRange[iPar].second);
            }
        }

        for (std::size_t iPar = 0; iPar < points->size(); ++iPar) {
            posteriorAverage[iPar] /= norm;
            posteriorRms[iPar] /= norm;
            posteriorRms[iPar]
                = posteriorRms[iPar]
                - posteriorAverage[iPar]*posteriorAverage[iPar];
            if (posteriorRms[iPar] > 0.0) {
                posteriorRms[iPar] = std::sqrt(posteriorRms[iPar]);
            }
            else {
                posteriorRms[iPar] = 0.0;
            }
            posteriorRange[iPar].second
                += 1E-8*(posteriorRange[iPar].second
                          -posteriorRange[iPar].first);
        }
    } while (false);

    // Initialize posterior parameter distribution histograms. Do this by
    // looping over the sets, and then the parameters within the sets.  It
    // could be done by looping over the gPar<blah> vectors directly, but this
    // eliminates odd special cases.
    std::vector<TH1D*> posteriorParamPlots(nTotal);
    for (std::size_t iSet = 0; iSet < gSetName.size(); ++iSet) {
        for (int iPar = gSetOffset[iSet];
             iPar < gSetOffset[iSet]+gSetCount[iSet]; ++iPar) {

            // Name of current parameter
            std::ostringstream str;
            str << "Par#" << std::setfill('0') << std::setw(3) << iPar;
            std::ostringstream histString;
            histString << "Par#" << std::setfill('0') << std::setw(3) << iPar
                       << "_" << gSetName[iSet]
                       << "_" << gParName[iPar];
            std::string histName = histString.str();
            std::string histTitle
                = gSetName[iSet] + "/" + gParName[iPar] + ": " + str.str();

            double halfRange = 5*posteriorRms[iPar];
            halfRange = std::max(0.0001,halfRange);
            double minRange = posteriorAverage[iPar]-halfRange;
            minRange = std::max(minRange,gParMinimum[iPar]);
            minRange = std::max(minRange,posteriorRange[iPar].first);
            double maxRange = posteriorAverage[iPar]+halfRange;
            maxRange = std::min(maxRange,gParMaximum[iPar]);
            maxRange = std::min(maxRange,posteriorRange[iPar].second);

            std::cout << "Construct " << iSet << "/" << iPar
                      << " [" << gSetName[iSet] << "]"
                      << " <" << gParName[iPar] << ">"
                      << std::endl
                      << "    prior: " << gParPrior[iPar]
                      << " +/- " << gParSigma[iPar]
                      << " [" << gParMinimum[iPar]
                      << " -- " << gParMaximum[iPar] << "]"
                      << std::endl
                      << "    post: " << posteriorAverage[iPar]
                      << " +/- " << posteriorRms[iPar]
                      << " [" << posteriorRange[iPar].first
                      << " -- " <<  posteriorRange[iPar].second << "]"
                      << std::endl
                      << "    plot range min: " << minRange
                      << " max: " << maxRange
                      << ((gParFixed[iPar] or not gParEnabled[iPar]) ?
                          " unused:" : "")
                      << ((gParFixed[iPar]) ? " fixed" : "")
                      << ((not gParEnabled[iPar]) ? " disabled" : "")
                      << std::endl;

            posteriorParamPlots[iPar] = nullptr;
            if (gParFixed[iPar]) continue;
            if (not gParEnabled[iPar]) continue;

            if (not std::isfinite(gParPrior[iPar])
                or not std::isfinite(gParSigma[iPar])
                or not std::isfinite(halfRange)
                or not std::isfinite(minRange)
                or not std::isfinite(maxRange)) {
                std::cout << "           Skipping for non finite values"
                          << std::endl;
                continue;
            }

            posteriorParamPlots[iPar]
                = new TH1D(histName.c_str(),histTitle.c_str(),
                           100, minRange, maxRange);

	} // End of current set
    } // End of all sets

    outputFile->cd();
    gCanvas = new TCanvas("canvas", "canvas");
    gCanvas->cd();
    mcmcTree->Draw("LogLikelihood");
    gCanvas->Update();
    auto summaryLikelihood
        = (TH1F*) gPad->GetPrimitive("htemp")->Clone("SummaryLogLikelihood");
    summaryLikelihood->Write();
    PrintDrawnCanvas("Summary_LogLikelihood",3);
    double llhAvg = summaryLikelihood->GetMean();
    double llhRMS = summaryLikelihood->GetRMS();

    gCanvas = new TCanvas("canvas", "canvas");
    gCanvas->cd();
    mcmcTree->Draw("StepRMS");
    gCanvas->Update();
    auto summaryStepRMS
        = (TH1F*) gPad->GetPrimitive("htemp")->Clone("SummaryStepRMS");
    PrintDrawnCanvas("Summary_StepRMS",3);
    summaryStepRMS->Write();

    gCanvas = new TCanvas("canvas", "canvas");
    gCanvas->cd();
    {
        int xSteps = std::min(nPostEntries/10000,200);
        std::ostringstream plt;
        plt << "LogLikelihood:Entry$"
            << ">>SummaryLogLikelihoodVsStep"
            << "(" << xSteps << "," << 0.0 << "," << (double) nPostEntries
            << "," << 100 << "," << llhAvg-5*llhRMS << "," << llhAvg+5*llhRMS
            << ")";
        std::cout << "Draw " << plt.str() << std::endl;
        mcmcTree->Draw(plt.str().c_str());
    }

    auto summaryLikelihoodStep
        = (TH1F*) gDirectory->Get("SummaryLogLikelihoodVsStep");
    summaryLikelihoodStep->SetTitle("LogLikelihood vs Entry");
    summaryLikelihoodStep->Draw("colz");
    gCanvas->Update();
    PrintDrawnCanvas("Summary_LogLikelihoodVsStep",3);
    summaryLikelihoodStep->Write();

    gCanvas = new TCanvas("canvas", "canvas",3);
    gCanvas->cd();
    mcmcTree->Draw("StepRMS:TotalSteps");
    gCanvas->Update();
    auto summaryStepRMSTotalSteps
        = (TH1F*) gPad->GetPrimitive("Graph")->Clone("SummaryStepRMSVsStep");
    PrintDrawnCanvas("Summary_StepRMSVsStep");
    summaryStepRMSTotalSteps->Write();

    gCanvas = new TCanvas("canvas", "canvas",3);
    gCanvas->cd();
    mcmcTree->Draw("AdaptiveCovarianceTrace:TotalSteps");
    gCanvas->Update();
    auto summaryCovarianceTotalSteps
        = (TH1F*) gPad->GetPrimitive("Graph")->Clone(
            "SummaryCovarianceTraceVsStep");
    PrintDrawnCanvas("Summary_CovarianceTraceVsStep",3);
    summaryCovarianceTotalSteps->Write();

    double effectiveSampleSize = 0.0;
    double eventWeight = 0.0;

    // Number of bins in the autocorrelation plots
    int autoBins = std::min(200,maxAuto);

    gPosteriorAutocorrelationAvg
        = new TProfile("Summary_AvgAutocorrelation",
                       "Average autocorrelations",
                       autoBins,0.0,maxAuto,"s");
    gPosteriorAutocorrelationHist
        = new TH2D("Summary_HistAutocorrelation",
                   "Distribution of autocorrelations",
                   autoBins,0.0,maxAuto,
                   200,-1.0, 1.0);

    fillAutocorrelations(mcmcTree, first, maxAuto, autoBins);

    /////////////////////////////////////////////////////////////////
    // Calculate and fill the average autocorrelations based on collected
    // data.
    /////////////////////////////////////////////////////////////////
    for (std::size_t iAcc = 0; iAcc < posteriorParamPlots.size(); ++iAcc) {
        // Skip parameters that will not be in the summary plots.  These are
        // parameters that were fixed and have a zero autocorrelation.
        if (!gPosteriorAutocorrelation[iAcc]) {
            std::cout << "Skip plot: " << iAcc << std::endl;
            continue;
        }
        for (int k = 1;
             k<=gPosteriorAutocorrelation[iAcc]->GetNbinsX(); ++k) {
            double a = gPosteriorAutocorrelation[iAcc]->GetBinContent(k);
            double c = gPosteriorAutocorrelation[iAcc]->GetBinCenter(k);
            gPosteriorAutocorrelationAvg->Fill(c,a);
            gPosteriorAutocorrelationHist->Fill(c,a);
        }
    }

    // Estimate the effective sample size.  The effective sample size is
    //
    // N / R or for finite samples N/Rhat
    //
    // R is ( 1 + sum(-inf,inf, autocorr)) or ( 1 + 2sum(0,inf, autocorr)) for
    // a symmetric function.  Rhat is a statistical quantity, but this is
    // assuming that we can estimate it perfectly using our chain (biased for
    // "odd" chains, but "good enough" for large well-mixed chains).
    //
    // N is the total number of samples in the sample
    //
    // Markov Chain Monte Carlo in Practice: A Roundtable Discussion. Robert
    // E. Kass, Bradley P. Carlin, Andrew Gelman and Radford M. Neal. The
    // American Statistician. Vol. 52, No. 2 (May, 1998), pp. 93-100
    //
    // Stochlastic Simulation, B. D. Ripley, ISBN: 978-0471818847 (1987)
    //
    // This entire formalism is built around the idea that a function is has a
    // well behaved autocorrelation, but there are entire classes of functions
    // that do not!
    for (double x = 1.0;
         x < gPosteriorAutocorrelationAvg->GetBinCenter(
             gPosteriorAutocorrelationAvg->GetNbinsX()-1);
         x += 1.0) {
        double r = gPosteriorAutocorrelationAvg->Interpolate(x);
        int b = gPosteriorAutocorrelationAvg->GetBin(x);
        double e = gPosteriorAutocorrelationAvg->GetBinError(x);
        double s = std::abs(r/e)/std::sqrt(2.0);
        effectiveSampleSize += std::erf(s)*std::abs(r);
    }
    effectiveSampleSize = steps/(1.0 + 2.0*std::abs(effectiveSampleSize));
    eventWeight = effectiveSampleSize/steps;

    std::vector<double> ess;
    for (std::size_t iPar = 0; iPar<gPosteriorAutocorrelation.size(); ++iPar) {
        ess.push_back(0.0);
        if (!gPosteriorAutocorrelation[iPar]) continue;
        for (double x = 1.0;
             x < gPosteriorAutocorrelation[iPar]->GetBinCenter(
                 gPosteriorAutocorrelation[iPar]->GetNbinsX()-1);
             x += 1.0) {
            double r = gPosteriorAutocorrelation[iPar]->Interpolate(x);
            int b = gPosteriorAutocorrelation[iPar]->GetBin(x);
            double e = gPosteriorAutocorrelation[iPar]->GetBinError(x);
            double s = std::abs(r/e)/std::sqrt(2.0);
            ess.back() += std::erf(s)*std::abs(r);
        }
        ess.back() = steps/(1.0 + 2.0*ess.back());
    }

    std::cout << "Effective sample size:    " << effectiveSampleSize
              << std::endl;
    std::cout << "Event weight:             " << eventWeight
              << std::endl;

    // Plot the average autocorrelation
    if (gPosteriorAutocorrelationAvg) {
        // Name of current parameter
        std::string histName{gPosteriorAutocorrelationAvg->GetName()};
        gCanvas->cd();
        gPosteriorAutocorrelationAvg->Draw();
        gPad->Update();
        PrintDrawnCanvas(histName,3);
    }

    // Plot the distribution of autocorrelations
    if (gPosteriorAutocorrelationHist) {
        // Name of current parameter
        std::string histName{gPosteriorAutocorrelationHist->GetName()};
        gCanvas->cd();
        gPosteriorAutocorrelationHist->Draw("zcol");
        gPad->SetLogz(true);
        gPad->Update();
        PrintDrawnCanvas(histName,3);
        gPad->SetLogz(false);
    }

    std::vector<TGraph*> posteriorPointsTrace;

    do {
        std::vector<float> *points =0;
        std::vector<double> *accepted =0;
        mcmcTree->SetBranchAddress("Points",&points);
        mcmcTree->GetEntry(0);

        if (points->size() != gParName.size()) {
            std::cout << "Problem with input data"
                      << " points " << points->size()
                      << " names " << gParName.size()
                      << std::endl;
            throw;
        }

        std::cout << "Entries " << nPostEntries << std::endl;
        for (int i = first; i < nPostEntries; i += stride) {
            mcmcTree->GetEntry(i);
            if ((printCounter++%10000) == 0) {
                std::cout << "Fill parameters with entry " << i << std::endl;
            }
            for (std::size_t iPar = 0; iPar < points->size(); ++iPar) {
                double content = points->at(iPar);
                if (!posteriorParamPlots[iPar]) {
                    std::cout << "Missing parameter " << iPar
                              << std::endl;
                    continue;
                }
                if (not std::isfinite(ess[iPar])) {
                    std::cout << "Parameter " << iPar
                              << " ESS is not finite:" << ess[iPar]
                              << std::endl;
                    continue;
                }
                if (ess[iPar] <= 0.0) {
                    std::cout << "Parameter " << iPar
                              << " ESS is invalid: " << ess[iPar]
                              << std::endl;
                    continue;
                }
                double w = ess[iPar]/steps;
                posteriorParamPlots[iPar]->Fill(content,w);
            }
        }

        if (fillTraces) {
            // A vector of parameter traces (might not be filled).
            for (std::size_t i=0; i<points->size() and fillTraces; ++i) {
                std::ostringstream graphTitle;
                graphTitle << "Trace for accepted steps of "
                           << gParSetName[i] << "/" << gParName[i]
                           << " Par #" << std::setfill('0')
                           << std::setw(3)  << i;
                posteriorPointsTrace.push_back(new TGraph());
                posteriorPointsTrace.back()->SetTitle(
                    graphTitle.str().c_str());
            }

            // fill  the accepted traces
            int traceLength = 1000;
            int traceStride = (nPostEntries-first)/traceLength;
            if (traceStride < 1) traceStride = 1;
            int point = 0;
            for (int i = first; i < nPostEntries; i += traceStride) {
                mcmcTree->GetEntry(i);
                if ((printCounter++%10000) == 0) {
                    std::cout << "Fill traces with entry at " << i << std::endl;
                }
                for (std::size_t iAcc = 0; iAcc < points->size(); iAcc++) {
                    double content = points->at(iAcc);
                    posteriorPointsTrace[iAcc]->SetPoint(point, i,  content);
                }
                ++point;
            }
        }
    } while (false);

    /// Fix the posterial parameter histogram errors to account for the
    /// effective sample size.
    for (std::size_t i = 0;
         i < posteriorParamPlots.size() and plotParams; ++i) {
        if (!posteriorParamPlots[i]) continue;
        // The histograms are normalized so that the integral is the effective
        // sample size.
        double integral = posteriorParamPlots[i]->Integral();
        if (integral < 1) continue;
        for (int b = 1; b <= posteriorParamPlots[i]->GetNbinsX(); ++b) {
             double c = posteriorParamPlots[i]->GetBinContent(b);
             // Explicitly calculate the binomial uncertainty.
             double p = c/integral;
             double q = 1.0-p;
             double s = integral*p*q;
             if (not std::isfinite(s) or s < 1.0) s = 1.0;
             s = std::sqrt(s);
             posteriorParamPlots[i]->SetBinError(b,s);
         }
    }

    // Build the summary plots.
    std::vector<TH1D*> postfitSigma;
    std::vector<TH1D*> postfitValue;
    std::vector<TH1D*> prefitValue;
    std::vector<TH1D*> prefitLine;

    for (std::size_t iSet = 0; iSet < gSetName.size(); ++iSet) {
        int summaryBins = 0;
        // Count the number of bins in the summary plots
        for (int iPar = gSetOffset[iSet];
             iPar < gSetOffset[iSet]+gSetCount[iSet]; ++iPar) {
            if (!posteriorParamPlots[iPar]) continue;
            ++summaryBins;
        }
        if (summaryBins < 1) continue;

        // The posterior distributions in terms of prior sigmas
        postfitSigma.push_back(
            new TH1D((gSetName[iSet]+" Post Sigmas").c_str(),
                     (gSetName[iSet]+" (Post Fit Sigmas)").c_str(),
                     summaryBins, 0, summaryBins));
        // The posterior value distributions
        postfitValue.push_back(
            new TH1D((gSetName[iSet]+" Post Values").c_str(),
                     (gSetName[iSet]+ " (Postfit Values)").c_str(),
                     summaryBins, 0, summaryBins));
        // The prior value distributions
        prefitValue.push_back(
            new TH1D((gSetName[iSet] + "Pre Value").c_str(),
                     (gSetName[iSet] + " (Prefit Values)").c_str(),
                     summaryBins, 0, summaryBins));
        // A copy of prefitValue so that the central value line can be easily
        // drawn.
        prefitLine.push_back(
            new TH1D((gSetName[iSet] + "Pre Line").c_str(),
                     (gSetName[iSet] + " (Prefit Line)").c_str(),
                     summaryBins, 0, summaryBins));

        int nextBin = 0;
        for (int iPar = gSetOffset[iSet];
             iPar < gSetOffset[iSet]+gSetCount[iSet]; ++iPar) {
            if (!posteriorParamPlots[iPar]) continue;
            ++nextBin;

            // Get Posterior content
            std::string binName(posteriorParamPlots[iPar]->GetTitle());
            double mean = posteriorParamPlots[iPar]->GetMean();
            double rms = posteriorParamPlots[iPar]->GetRMS();

            // Fill posterior content
            postfitValue.back()->SetBinContent( nextBin, mean);
            postfitValue.back()->SetBinError( nextBin , rms);

            if (gParSigma[iPar] > 0.0) {
                postfitSigma.back()->SetBinContent(
                    nextBin,
                    (mean-gParPrior[iPar])/gParSigma[iPar]);
                postfitSigma.back()->SetBinError(
                    nextBin,
                    rms/gParSigma[iPar]);
            }

            // Fill prior content
            prefitValue.back()->SetBinContent( nextBin, gParPrior[iPar]);
            prefitValue.back()->SetBinError( nextBin, gParSigma[iPar]);
            prefitLine.back()->SetBinContent( nextBin, gParPrior[iPar]);

            // Labels
            postfitSigma.back()->GetXaxis()->SetBinLabel( nextBin,
                                                          binName.c_str());
            postfitValue.back()->GetXaxis()->SetBinLabel( nextBin,
                                                          binName.c_str());
            prefitValue.back()->GetXaxis()->SetBinLabel( nextBin,
                                                         binName.c_str());
        }
    }

    outputFile->cd("/Summary");

    // Plot posterior vs. prior distribution for each set
    for (std::size_t iSet = 0; iSet < gSetName.size(); ++iSet) {
        postfitSigma[iSet]->Write();
        postfitValue[iSet]->Write();
        prefitValue[iSet]->Write();

        gCanvas = new TCanvas("canvas", "canvas");
        gCanvas->cd();
        postfitValue[iSet]->GetXaxis()->SetLabelSize(0.03);
        postfitValue[iSet]->GetXaxis()->LabelsOption("v");
        postfitValue[iSet]->SetMarkerColor(9);
        postfitValue[iSet]->SetLineColor(9);
        postfitValue[iSet]->SetMarkerStyle(kFullDotLarge);
        postfitValue[iSet]->SetLineWidth(2);
        postfitValue[iSet]->SetFillColor(kRed-9);
        gCanvas->Update();

        prefitValue[iSet]->GetXaxis()->SetLabelSize(0.03);
        prefitValue[iSet]->GetXaxis()->LabelsOption("v");
        prefitValue[iSet]->SetMarkerColor(kRed-3);
        prefitValue[iSet]->SetFillColor(kRed-9);
        prefitValue[iSet]->SetMarkerStyle(kFullDotLarge);
        prefitValue[iSet]->SetMarkerSize(0);
        prefitValue[iSet]->Draw("E2");
        prefitLine[iSet]->SetLineColor(kRed-3);
        prefitLine[iSet]->Draw("same");
        gCanvas->Update();

        gStyle->SetOptStat(0);
        gCanvas->SetGrid();
        gCanvas->SetBottomMargin(0.20);
        postfitValue[iSet]->GetYaxis()->SetRangeUser(-7,12);
        postfitValue[iSet]->Draw("E1 X0 same");
        gCanvas->Update();
        PrintDrawnCanvas("Summary_" + gSetName[iSet] + "_values",3);

        gCanvas = new TCanvas("canvas", "canvas");
        gCanvas->cd();
        postfitSigma[iSet]->GetXaxis()->SetLabelSize(0.03);
        postfitSigma[iSet]->GetXaxis()->LabelsOption("v");
        postfitSigma[iSet]->GetYaxis()->SetRangeUser(-3,3);
        postfitSigma[iSet]->SetMarkerColor(9);
        postfitSigma[iSet]->SetLineColor(9);
        postfitSigma[iSet]->SetMarkerStyle(kFullDotLarge);
        postfitSigma[iSet]->SetLineWidth(2);
        postfitSigma[iSet]->SetFillColor(kRed-9);
        gCanvas->Update();

        gStyle->SetOptStat(0);
        gCanvas->SetGrid();
        gCanvas->SetBottomMargin(0.20);
        gCanvas->Update();
        postfitSigma[iSet]->Draw("e1 X0");
        gCanvas->Update();
        PrintDrawnCanvas("Summary_" + gSetName[iSet] + "_sigmas",3);
    }

    outputFile->cd("/Parameters");
    // Plot 1d parameter distributions
    for (std::size_t i = 0;
         i < posteriorParamPlots.size() and plotParams; ++i) {
        if (!posteriorParamPlots[i]) continue;

        // Name of current parameter
        std::string histName{posteriorParamPlots[i]->GetName()};
        std::cout << "Print " << histName << std::endl;

        // Plot 1D distribution
        gCanvas->cd();
        gStyle->SetOptStat("iksrmeou");
        posteriorParamPlots[i]->SetMarkerColor(kBlue);
        posteriorParamPlots[i]->SetMarkerStyle(kFullDotLarge);
        posteriorParamPlots[i]->SetFillColor(kRed-9);
        posteriorParamPlots[i]->Draw("E2");
        gPad->Update();
        posteriorParamPlots[i]->Write();
        PrintDrawnCanvas(histName);
        gStyle->SetOptStat(0);
    }

    // Plot 1d parameter autocorrelations
    outputFile->cd("/Autocorrelations");
    for (std::size_t i = 0;
         i < gPosteriorAutocorrelation.size() and plotAuto; ++i) {
        if (!gPosteriorAutocorrelation[i]) continue;

        // Name of current parameter
        std::string histName{gPosteriorAutocorrelation[i]->GetName()};
        std::cout << "Autocorrelation " << histName << std::endl;

        // Plot 1D distribution
        gCanvas->cd();
        gPosteriorAutocorrelation[i]->Draw("E1 X0");
        gPad->Update();
        gPosteriorAutocorrelation[i]->Write();
        PrintDrawnCanvas(histName);
    }

    // Plot points traces
    outputFile->cd("/Traces");
    for (std::size_t i = 0; i < posteriorPointsTrace.size(); ++i) {
        if (!posteriorPointsTrace[i]) continue;

        std::ostringstream histName;
        histName << "Trace"
                 << "_" << gParSetName[i]
                 << "_" << gParName[i]
                 << "_Par#" << std::setw(3) << std::setfill('0') << i;

        std::cout << "Trace " << histName.str() << std::endl;

        // Plot trace
        gCanvas->cd();
        posteriorPointsTrace[i]->Draw("AC");
        gPad->Update();
        posteriorPointsTrace[i]->Write();
        if (plotTraces) PrintDrawnCanvas(histName.str());
    }

    std::cout << "Effective sample size:" << effectiveSampleSize << std::endl;
    std::cout << "Event weight:             " << eventWeight << std::endl;

}

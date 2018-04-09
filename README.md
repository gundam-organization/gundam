# likelihoodFitter

/*! \mainpage Overview

 \section intro_sec Introduction
This is the first commit and first documentation for the xsLLhFitter package. This file is designed to be read by doxygen and converted into documentation. To do this you can run doxygen on Doxyfile.dxy. In the future I'll use the lancs doxygen server.

For questions/comments/complaints/rants drop me an email: s.dolan@physics.ox.ac.uk

The idea of this package is to provide a tool that is as generic as possible to allow xsec analysers to try xsec extraction using a likelihood fit. The current state of the code and documentation is very much work in progress and is really not ready for general use. However, if you're desperate or fancy helping with development then read on!

This document is currently all about the code for the fitter. For anything related to the principles behind the fitter, have a read of TN214, TN261, TN263 and TN287.

 \section install_sec Installation

 \subsection install_gmake Hand-spun gmake
Before doing anything make sure you have root sourced. Next check the fitter out into some working area (cvs co xsLLhFitter -assuming you have your CVSROOT set to the ND280 repo). Change directory to the root of the fitter (/some/path/xsLLhFitter) and do the following:

$ make init

This creates softlinks in the ./include folder

$ make all

This will compile the fitter, leaving an exe called ccqefit.exe in /fitter/root/dir/bin

You can use make clean if you run into problems and need to start again

 \subsection install_cmake CMake

The only external dependency of the package is ROOT. Set up the ROOT environment
before attempting to build by:

    $ source /path/to/ROOT/bin/thisroot.sh

Then source the package setup script.

    $ source /path/to/xsLLhFitter/setup.sh

  this may take a few minutes to complete as it will check for CMake and download
  and build the one from ND280Repo if it doesn't exist on your system.
  The first time this script is run it will notify you that it cannot find the
  build setup script, this is normal.

To build:

    [1] $ mkdir build; cd build;
    [2] $ cmake ../
    [3] $ make install -j
    [4] $ source $(uname)/setup.sh

and you should be done. The default build is `DEBUG`, which compiles the libraries
statically and includes debugging symbols. If [2] is replaced by
`cmake -DCMAKE_BUILD_TYPE=RELEASE ../` then debugging symbols will be turned off,
some compiler optimisations turned on and libraries will be build, and linked,
as shared objects.

For future setups, the setup.sh in the package root will source the one installed
alongside the executables and libraries and is the only setup this package needs.
It will also source the ROOT installation used at configure time, so manual re-sourcing
or setup script wrapping to include ROOT is not neccessary.

 \subsubsection install_cmake_notes CMake Notes:

  * Unfortunately the default ROOT CMake script is provided broken on a number of
    versions of ROOT. We will provide patches for the latest ND280 supported ROOT
    version, but there are numerous ROOTTalk posts about the specific problems
    and fixes for each problem.

 \section gettingstarted_sec Getting Started

I've provided enough sample input files for you to have a go at running the fitter with detector and most model systematics. This section will walk you through how to do this and explain the basic general use of the fitter along the way.  This section will not really cover generating your own inputs for the fitter or plotting the results nicely.

The example is an Asimov (fake data==MC) 1D fit to extract a CC0Pi+Np(N>0) differential xsec in dpt bins (stored as D2 in the binning).

\subsection inpputs_subsec Inputs

The most important file is /src/ccqefit.cc. This is equivalent (although currently nowhere near as general) as the macros you may use to run xsTools. It takes inputs from you needed for fitting (e.g where are your data/MC/binning/covariance matrices/splines, what are your samples/POTs/regularisations etc.) and then sets up and runs a MINUIT based likelihood fitter.

The first couple of lines of ccqefit takes the location of the MC (fsel), fake data, the covariance matrices for detector and flux systematics, the binning and the default output file. It then takes the POT of the MC and fake data (so the MC can be scaled appropriately), the random seed and the regularisation strength.

As it stands, every time you change any of these in ccqefit.cc you'll need to recompile (make from root dir) but this headache is avoided by also allowing most of these options to be taken as command line arguments. Run the fitter as ccqefit.exe -h to see the tags you need to specify.

Now we've chosen some inputs to the fitter the code loads these in. There're a couple of bits of hard coding here that are described in the comments. If you're using the example files or currently don't care about systematics then you don't need to change any of this. If you do want to add your own systematics and the comments in the code don't make sense then let me know and I can help.

\subsection samples_subsec Samples

This takes up to around line 450 where we specify our samples. A sample is an event selection that characterises a particular final state topology which will have a particular detector acceptance. For example if you’re looking for just one muon from an FGD vertex then you’d probably want a sample where you see the muon in the TPC and a sample where you see the muon in the FGD. You can’t put these together since the smearing matrices (map from reco->truth of some variable) and efficiency corrections will be quite different for each. Samples can also be used to include sidebands (AKA control regions) in the fit. Samples must always be mutually exclusive so we don’t double count events.

Samples are stored in the class AnySample which requires a number, a name and binning edges (derived from your binning input). The number should be the cutBranch leaf stored in the input tree (more on this later) which would normally correspond to the HL2 branch the sample comes from.

There is no need to specify what sample corresponds to what type of signal or background since in the fitter all samples are treated identically. The fitter will `know’ what type of events the sample characterises since the sample will be effected much more by reweighting some background (or signal) than others.

\subsection fittersettigns_subsec Setting up the fitter

Below the section on samples (c. L500) the mode of the fitter are chosen. Parameters to fit must be pushed_back into the vector of AnaFitParameters called fitpara. In the ccqefit example all but the fit parameters (rather than the systematic –or nuisance- parameters) are commented out.

If you want to uncomment the detector, xsec and pion FSI (not nucleon FSI!) parameters you should be able to (but the fit will take much longer).

On line 600 the frequency of saving the fitter status in the output file is chosen (xsecfit.SetSaveMode). Here 1 corresponds to save once in every 10000 MINUIT calls. Then just below the fitter is finally set going (xsecfit.Fit). The fitter can be ran in one of three modes. Mode 3 is for extracting fit results whilst mode one is for making a stats only pulls study whilst mode 1 is for systematic pull studies. In the current incarnation of the fitter modes 1 and 3 are untested.

\subsection outsideccqefit_subsec Settings beyond ccqefit.cc

To configure the fitter for a new analysis the vast majority of changes only need to be to ccqefit. In the long run the ONLY changes should be to ccqefit. However, at the moment this isn’t quite the case.

The most important non ccqefit setting is the signal definition i.e. what it is that you’re actually trying to measure. This is set in /fitparam/src/FitParameters.cc around line 100. There is a comment in the code above the if statement and an example. Typically you’ll set a signal using the ‘reaction` variable. This will typically be a category from HL2, in the example case it’s the mectopology category.  You can also add phase space cuts to your signal definition.

\subsection runningfitter_subsec Running the fitter

You’re now ready to run the fitter! Make sure everything is compiled and head to /bin. Then run ./ccqefit.exe and watch far too much output appear on your screen. An output file will be produced as specified in ccqefit (or by a –o tag to the exe if you prefer). There will also be an output txt file but ignore this for the moment, this is used for regularisation studies. The fit result (parameter values and errors) can be found by scrolling up a little, past the large correlation matrices printed to screen.

The output root file first contains the distribution of events from each sample and reaction (elements of your category) in your chosen binning (reco and true) from the MC. It also contains the distribution of events in “AnyBins” which is just the distribution in the global bin number of the fitter (i.e. 2D binning collapsed into 1D). In this example only D2 is used (it’s dpt) D1 has only one bin that covers it’s entire phase space. In this way we use the 2D fitter to do a 1D fit.

After this there’re histograms called things like evhist_sam0_iter0_... These are binned in global bin number and contain the data mc and fit (pred) results at iter 0. Since MINUIT hasn’t been called yet pred==MC and since this is an Asimov fit MC==data so these will all be the same.

Under this is the chi2 reported by MINUIT and the fit result (the fitted value of each parameter) and error.

Next is once again the distribution of events in each sample and reaction but this time the fitted result rather than what was in the MC.

Finally the ``evhist_pred’’ histograms appear again but this time for the final MINUIT iteration (they’re actually listed twice, once with the iteration number and once with finalIter but these should be identical – this is just needed for my xsec drawing macro to work). If the fit is working then these should be similar to the evhist_sam0_iter0_data from earlier but since this example is an Asimov fit nothing should have actually moved.

And that’s it. The fitter has ran and you’ve got some results out. How to plot these results in a way that makes them easy to interpret and and how to create your own inputs for the fitter will be covered in the next sections.


\section plotting_sec Drawing the results
There’s a macro to do this in /macros called calcXsecWithErrors for the example files this will work out the box. To make the overall result work for a new analysis you should only have to change the signal definition around line 144 and 149 and the pot of your MC on line 185. This is assuming you have less than 10 samples

The macro is just for testing as it stands and is certainly not a very good drawing tool.

The macro can be compiled in root (.L calcXsecWithErrors.cc+) and ran with calcXsecWithErrors(…). The arguments are the MC file location (e.g. inputs/NeutAirInputTree.root), the fit result file (e.g. bin/outfilename.root), the fake data filename (same as the MC in the example), the output root file name to store the plots in and finally the POT ratio between data and MC (1 in the example).

The output file will contain some histograms and canvases. If there’s an _T then an efficiency correction has been applied (with the exclusion of xsecCanv since a cross section will always have to have had an efficiency correction applied). In general red is the fit result, green is fake data and blue is MC.

Signal Comp (or SigComp_T) contains the truth result of the fit. How close the red is to the green (regardless of the blue) is a good measure of how well the fit is doing. Since the example is an Asimov fit all three should look the same.

All Sample Comp contains the reco result of the fit. The green is what is actually measured in the detector. This is what the fit actually sees and fits so here the red should really be close to the green. If it’s not then there’s either something wrong with the fitter or your binning is too small compared to detector smearing effects.

AllTruthAndRec shows the previous two canvases together.

xsecCanv shows the overall flux integrated cross section extrapolated from the fit. This is just a scale factor and normalisation of bin width away from SigComp_T.

The error bars are just the propagated error bars from the fit results. If you included all systematics, validated that your error are Gaussian and ran MINUIT in HESSE or MINOS mode (not documented here yet) then the error bars on the cross section are only missing the additional errors due to the efficiency correction.



\section inputs_sec Inputs to the fitter
\subsection treeconvert_subsec treeConvert.cc
This very simple macro converts HL2 microtrees into the format needed for the fitter to work whilst allowing reweighting events and making additional cuts to test the fitter. The macro stores the analysis variables along with the proton and muon kinematic variables. If you want to look at pions then you’ll need to either replace the proton variables or add more information. This has a lot of hard coding and will need to be modified for any analysis that doesn’t use the numuCCZeroPi HL2 package. You’ll need to change the nbranches, accumToCut, when to use mom by range and, if you want to reweight the signal, the signal definition.

treeConvert.cc can be compiled with root (.L treeConvert.cc+) and ran as follows:
treeConvert(“path/to/HL2/microtree.root”, “default”, “truth”, “/path/to/output/tree.root”,  1.0, 0, 0, “varNameInHL2defaultTreeToStoreAsD1Rec”, “varNameInHL2defaultTreeToStoreAsD1True”, “varNameInHL2truthTreeToStoreAsD1TrueT” , “varNameInHL2defaultTreeToStoreAsD2Rec”, “varNameInHL2defaultTreeToStoreAsD2True”, “varNameInHL2truthTreeToStoreAsD2TrueT”)

More instructions will be in future update.

\subsection calcdetcov_subsec calcCovMat.C
This macro (and the _fine version) are needed to calculate detector covariance matrices from HL2. If you have uniform binning then HL2 drawing tools can do this quicker and better. If you don’t have uniform binning I don’t think it can. If I’m wrong then please let me know! To make this work you should only have to change the binning section of the code. You’ll also need to tell ccqefit.cc where to find the binning for the coarse cov mat binning.

All the macro does is calculate the covariance after throwing detector systematics in some chosen binning. The result is a covariance matrix (one with absolute values and one with relative values).

More instructions will be in future update.

\subsection calcdetcov_subsec genResponse
The genResponce_dpt macro and script to run it uses xsTools neat reweighting functions to calculate response functions (pretty much the same as splines) for the fitter. To use this you’ll need to edit the macro to use your variable and your binning in addition to pointing the macro at your nd280 flux file. You’ll also have to have xsTools libraries sourced and used the appropriate xsTools macro to create reweight files.  You’ll have to edit the for loop to cover your samples (/HL2 cut branches) and relevant analysis category.

You’ll also need to edit the script to point to your HL2 microtree you want to reweight and the path to the reweight files as well as only looping over the correct samples (topo) and category (reac).

This will be made more general in the future. If you need it now and can’t understand what’s been done then drop me an email.

More instructions will be in future update.

\section adapting_sec Adapting the fitter to your analysis

You’ll need to tell the fitter about your chosen reaction category if it isn’t mectopology. This is done in anaevents/src/AnySample. You should change nreac and names in the method GetSampleBreakdown to describe your category.

If you are adding a different set of xsec or FSI systematics you’ll need to add the covariance matrices for them into ccqefit.cc but this also requires some hard coding outside of ccqefit.cc. In this case have a look at fitparam/src/XsecParameters.cc or fitparam/src/FSIParameters.cc. You’ll need to specify the steps and upper and lower limits of each parameter whilst also editing the for loops in StoreResponceFunctions to suit your samples and reactions.

 */

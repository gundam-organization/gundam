## T2KReWeight

The cross-section parameters for the fit require some method of changing the weight of an event given a change in the parameter. This is achieved through a series of spline functions generated with the T2KReWeight framework. The Super-xsllh Fitter expects a set of these splines for use with the cross-section parameters, and requires the splines to be in the correct format (unless the code is modified).

The `reweight/` directory contains the code used to run T2KReWeight and generate spline files in the format the fit expects. Examples of required input to the code are also available in the directory. Currently, the Super-xsllh repository will not build or link to T2KReWeight, so the reweight code must be copied to the user's installation of T2KReWeight. A Makefile is provided to compile the reweight code, and it requires the C++11 standard as written.

One warning, the reweight code is currently hardcoded to produce splines binned in topology, reaction, and true muon momentum and angle. The event selection and variables are also hardcoded. The reweight code definitely needs to be looked at before using for a given analysis!

# Costas Andreopoulos <costas.andreopoulos \at stfc.ac.uk>

SHELL = /bin/sh
NAME = all
MAKEFILE = Makefile

# Include machine specific flags and locations (inc. files & libs)
#
include $(T2KREWEIGHT)/make/Make.include
#include $(ROOTSYS)/include/Make.incude


#all: genWeights_2013a genWeightsFromGHEPTree genWeightsFromGRooTracker_FluxTuningexample genWeightsFromNRooTracker_NEUTexample genWeightsFromSK_NIWGexample
#all: genWeightsFromNRooTracker_NEUTexample genWeightsFromNRooTracker_P0DTPC1 FSIgenWeightsFromNRooTracker_P0DTPC1 FSIAllgenWeightsFromNRooTracker_P0DTPC1 CovBANFFgenWeightsFromNRooTracker_P0DTPC1 BANFFCovFluxgenWeightsFromNRooTracker_P0DTPC1 QECorrelationgenWeightsFromNRooTracker_P0DTPC1 testsubBANFFCovFluxgenWeightsFromNRooTracker_P0DTPC1
all:
# Please only add new apps to be compiled by make all if you have checked 
# they still compile with following config:
# ./configure --disable-jnubeam --disable-neut --disable-genie --disable-oaanalysis;
# throwParmsFromJnuBeam 

processT2KrwOut: FORCE
	@echo -e "\n\n***** Compiling processT2KrwOut\n"
	$(CXX) $(CXXFLAGS) -c processT2KrwOut.cxx $(INCLUDES)
	$(LD) $(LDFLAGS) processT2KrwOut.o $(LIBRARIES) -o processT2KrwOut.exe  

genSplines: FORCE
	@echo -e "\n\n***** Compiling genSplines\n"
	$(CXX) $(CXXFLAGS) -c genSplines.cxx $(INCLUDES)
	$(LD) $(LDFLAGS) genSplines.o $(LIBRARIES) -o genSplines.exe  

genSplinesAir: FORCE
	@echo -e "\n\n***** Compiling genSplinesAir\n"
	$(CXX) $(CXXFLAGS) -c genSplinesAir.cxx $(INCLUDES)
	$(LD) $(LDFLAGS) genSplinesAir.o $(LIBRARIES) -o genSplinesAir.exe  





purge: FORCE
	$(RM) *.o *~ core 

clean: FORCE
	$(RM) *.o *~ core
	$(RM) *.exe

FORCE:

# DO NOT DELETE

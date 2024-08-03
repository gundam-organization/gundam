# Costas Andreopoulos <costas.andreopoulos \at stfc.ac.uk>

SHELL = /bin/sh
NAME = all
MAKEFILE = Makefile
DEFS = 

# Include machine specific flags and locations (inc. files & libs)
#
include $(T2KREWEIGHT)/make/Make.include

all: xsllh_weights xsllh_splines xsllh_validate

xsllh_weights:
	@echo -e "\n Compiling xsllh_response_functions\n"
	$(CXX) $(CXXFLAGS) $(DEFS) -std=c++11 -c xsllh_response_functions.cxx $(INCLUDES)
	$(LD) $(LDFLAGS) xsllh_response_functions.o $(LIBRARIES) -o xsllh_response_functions.exe

xsllh_splines:
	@echo -e "\n Compiling xsllh_generate_splines\n"
	$(CXX) -std=c++11 -g `root-config --cflags` `root-config --libs` xsllh_generate_splines.cxx BinManager.cxx -o xsllh_generate_splines.exe

xsllh_validate:
	@echo -e "\n Compiling xsllh_validate_splines\n"
	$(CXX) -std=c++11 -g `root-config --cflags` `root-config --libs` xsllh_validate_splines.cxx BinManager.cxx XsecDial.cxx -o xsllh_validate_splines.exe

purge:
	$(RM) *.o *~ core

clean:
	$(RM) *.o *~ core
	$(RM) *.exe

FORCE:

# DO NOT DELETE




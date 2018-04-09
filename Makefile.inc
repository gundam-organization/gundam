TOP := $(dir $(lastword $(MAKEFILE_LIST)))

ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS   = $(shell root-config --libs)
ROOTGLIBS  = $(shell root-config --glibs)

CXX        = g++
CXXFLAGS   = -g -Wall
LD         = g++

CXXFLAGS  += $(ROOTCFLAGS) -I $(TOP)include/
LIBS       = $(ROOTLIBS) $(SYSLIBS) -lMinuit 
GLIBS      = $(ROOTGLIBS) $(SYSLIBS) -lMinuit

AR         = ar
ARFLAGS    = rcsv

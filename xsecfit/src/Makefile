include ../../Makefile.inc

OBJS  = $(patsubst %.cc, %.o, $(wildcard *.cc))	
OBJS += XsecFitterDict.o
ARXIV = libxsecfit.a

all: $(ARXIV)

$(ARXIV): $(OBJS) 
	$(AR) $(ARFLAGS) $@ $^

#make ROOT dictinary class
XsecFitterDict.o: XsecFitter.hh
	rm -fv XsecFitterDict.h XsecFitterDict.cc
	rootcint XsecFitterDict.cc -c -I$(TOP)include $< 
	$(CXX) $(CXXFLAGS) -c XsecFitterDict.cc -o $@

clean:
	rm -fv *.o XsecFitterDict.cc XsecFitterDict.h
	rm -fv $(ARXIV)

distclean: clean
	rm -fv *.*~

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

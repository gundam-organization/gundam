include ../Makefile.inc

OBJS  = $(patsubst %.cc, %.o, $(wildcard *.cc))

all: $(OBJS)

clean:
	rm -fv *.o

distclean: clean
	rm -fv *.*~

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

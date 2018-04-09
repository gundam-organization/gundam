include ../../Makefile.inc

OBJS  = $(patsubst %.cc, %.o, $(wildcard *.cc))
ARXIV = libutils.a

all: $(ARXIV)

$(ARXIV): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

clean:
	rm -fv *.o
	rm -fv $(ARXIV)

distclean: clean
	rm -fv *.*~

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@


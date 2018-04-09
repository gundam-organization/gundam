include Makefile.inc

# modules to compile
MODULES = anaevents fitparam utils \
          xsecfit

EXECS   = $(patsubst src/%.cc, bin/%.exe, $(wildcard src/*.cc))
ARXIVs  = $(foreach mod, $(MODULES), $(mod)/src/lib$(mod).a)

.PHONY: all, init, clean, distclean, buildmod

all: $(EXECS)

$(EXECS) : buildmod

# rule for building executables
bin/%.exe : src/%.cc
	$(CXX) $(CXXFLAGS) $< $(ARXIVs) -o $@ $(LIBS)

# build modules
buildmod:
	@for mod in $(MODULES);\
	do make -C $$mod/src; done

# run first to set-up includes
init:
	@ [ -d include ] || mkdir include
	@ [ -d bin ]     || mkdir bin
	@for mod in $(MODULES);\
	do \
	for file in $$mod/src/*.hh;\
	do ln -s ../$$file include/; done \
	done

# clean libs
clean:
	@for mod in $(MODULES);\
	do make clean -C $$mod/src; done
	make clean -C src/
	rm -fv $(EXECS)

# clean dist
distclean: clean
	@find include/ -type l -delete
	@for mod in $(MODULES);\
	do make distclean -C $$mod/src; done
	make distclean -C src/

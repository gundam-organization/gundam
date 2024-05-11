#!/bin/bash
#
# Setup the GUNDAM directory for development (or simply
# running the test).  This makes sure that the ROOT environment
# variables are set (using thisroot.sh) since that helps debugging.
#
#  gundam-build == Source ./cmake/scripts/gundam-build.sh which will conveniently
#           run cmake/make/make install from any place so that it's
#           really easy to recompile.
#
#  gundam-setup == Source this file.  You probably never have to use
#           this one.
#
# This setup script is not needed.  You can also do it by hand.  It's
# a usual cmake build, but you need to make sure root is
# "in the path".
#
# source thisroot.sh
# cd the-build-directory
# cmake -DCMAKE_INSTALL_PREFIX=the-install-directory the-gundam-directory 
# make
# make install

if [ "x${BASH_VERSION}" = "x" ]; then
    echo
    echo ERROR: Setup script requires bash.  The GUNDAM build can be hand
    echo configured by setting the environment variables.
    echo
    echo GUNDAM_ROOT   -- The root for the gundam source
    echo GUNDAM_BUILD  -- The location for the build
    echo GUNDAM_INSTALL -- The location to install gundam - defaults to build
    echo GUNDAM_JOBS    -- Number of jobs to use during build
    echo GUNDAM_CMAKE_DEFINES -- Location specific cmake arguments
    echo
    return
fi

echo Setup gundam for development and building

# Try to setup root.  ROOT installs thisroot.sh in the bin directory
# to setup the environment.  The first "thisroot.sh" in the path will
# define the root that is used.
if [ "x${ROOTSYS}" = x ]; then
    echo ROOT not available.  Make sure that you have sourced thisroot.sh
    echo before gundam is setup.  You will need to run thisroot.sh before you
    echo can build.
fi

# Find the root of the building area.
___gundam_root() {
    COUNT=5
    while true; do
	if [ -e ./cmake -a -d ./cmake -a -e ./cmake/scripts/gundam-build.sh ]; then
	    echo ${PWD}
	    return
	fi
	COUNT=$(expr ${COUNT} - 1)
	if [ ${COUNT} -lt 1 ]; then
	    echo invalid-directory
	    return
	fi
	cd ..
    done
}

export GUNDAM_ROOT
GUNDAM_ROOT=$(___gundam_root)
unset -f ___gundam_root

if [ ${GUNDAM_ROOT} = "invalid-directory" ]; then
    echo The gundam-setup.sh must be sourced in the GUNDAM directory tree
    return
fi

___gundam_target () {
    target="gundam"
    if [ ${#GUNDAM_COMPILER} -gt 0 ]; then
        echo Using ${GUNDAM_COMPILER}
        compler=${GUNDAM_COMPILER}
    elif which gcc >& /dev/null ; then
        compiler=gcc
    elif which clang >& /dev/null; then
        compiler=clang
    fi
    case ${compiler} in
        gcc)
            compiler_version=$(gcc -dumpversion)
            compiler_machine=$(gcc -dumpmachine)
            ;;
        clang)
            compiler_version=$(clang -dumpversion)
            compiler_machine=$(clang -dumpmachine)
            ;;
        *)
            compiler_version=unknown
            compiler_machine=$(uname -s)-$(uname -m)
            ;;
    esac
    target="${target}-${compiler}_${compiler_version}-${compiler_machine}"
    echo $target
}

export GUNDAM_TARGET
GUNDAM_TARGET=$(___gundam_target)
unset -f ___gundam_target

___path_prepend () {
    ___path_remove $1 $2
    eval export $1="$2:\$$1"
}
___path_remove ()  {
    export $1=$(eval echo -n \$$1 | \
	awk -v RS=: -v ORS=: '$0 != "'$2'"' | \
	sed 's/:$//'); 
}

___path_prepend PATH ${GUNDAM_ROOT}/${GUNDAM_TARGET}/bin
___path_prepend LD_LIBRARY_PATH ${GUNDAM_ROOT}/${GUNDAM_TARGET}/lib

unset -f ___path_prepend
unset -f ___path_remove

alias gundam-setup=". ${GUNDAM_ROOT}/cmake/scripts/gundam-setup.sh"
alias gundam-build="${GUNDAM_ROOT}/cmake/scripts/gundam-build.sh"

echo Source code: ${GUNDAM_ROOT}
echo Build target: ${GUNDAM_TARGET}
echo Define GUNDAM_BUILD to change the default build area
echo Define GUNDAM_INSTALL to change the default installation area
echo Defined gundam-setup to re-setup the GUNDAM package.
echo Defined gundam-build to build the the GUNDAM package.

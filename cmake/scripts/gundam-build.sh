#!/bin/bash
#
# Build GUNDAM using cmake.  This checks the environment, setups a
# compiler/machine dependent build directory, and figures out the
# installation directory.  It then runs cmake and make.
#
# gundam-build [force] [cmake] [clean] [help] [-D<cmake-define>]
#     force -- Force cmake to ignore the cache
#     cmake -- Don't compile the source (only run cmake).
#     clean -- Run clean the build area after running cmake (run make clean)
#     keep-going  -- Continue after compilation errors (add -k to make)
#     test  -- Run tests after the build
#     verbose -- Run make with verbose turned on.
#     help  -- This message
#
#     -D<CMAKE_DEFINE> -- Add a definition to the cmake command.
#
# Set the GUNDAM_JOBS environment variable (defaults to 1) to control
# the number of threads used during the compilation.
#
# Set the GUNDAM_CMAKE_DEFINES environment variable (defaults to "")
# to provide default cmake command line definitions.  This can be used
# to override the defaults set in cmake/options.cmake for a particular
# system.  e.g. A system that has GPU might want to use:
#
# `export GUNDAM_CMAKE_DEFINES="-DWITH_CUDA_LIB=ON"`

# Check that the source root directory is defined.
if [ ${#GUNDAM_ROOT} == 0 ]; then
    echo GUNDAM is not setup yet.
    echo You can also run cmake and make by hand...
    exit 1
fi

# Check to make sure that ROOT is setup.
if [ ${#ROOTSYS} == 0 ]; then
    echo ROOT is not setup yet.
    echo Be sure to source thisroot.sh for your version of root.
    exit 1
fi

# Set or get the build location.  If GUNDAM_BUILD is not set, then the
# build is in a machine specific subdirectory inside GUNDAM_ROOT
if [ ${#GUNDAM_BUILD} == 0 ]; then
    BUILD_LOCATION=${GUNDAM_ROOT}/${GUNDAM_TARGET}
    if [ ! -d ${BUILD_LOCATION} ]; then
        mkdir -p ${BUILD_LOCATION}
    fi
else
    BUILD_LOCATION=${GUNDAM_BUILD}
    echo GUNDAM will be built use specified location
    echo Build Location: ${BUILD_LOCATION}
    echo The location must be created before gundam can be built.
fi

# Make sure the build directory exists (safety check).
if [ ! -d ${BUILD_LOCATION} ]; then
    echo Unable to access build location at ${BUILD_LOCATION}
    exit 1
fi

# If GUNDAM_INSTALL is not set, then the installation is in the same
# directory as the build.
if [ ${#GUNDAM_INSTALL} == 0 ]; then
    GUNDAM_INSTALL=${BUILD_LOCATION}
fi

FORCE_CMAKE="no"
ONLY_CMAKE="no"
RUN_CLEAN="no"
RUN_TEST="no"
DEFINES=" -DCMAKE_INSTALL_PREFIX=${GUNDAM_INSTALL} "
DEFINES="${DEFINES} -DCMAKE_EXPORT_COMPILE_COMMANDS=1 "
DEFINES="${GUNDAM_CMAKE_DEFINES} ${DEFINES}"
if [ "x${GUNDAM_JOBS}" == x ]; then
    GUNDAM_JOBS=1
    echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    echo Threads: ${GUNDAM_JOBS} -- Override with GUNDAM_JOBS shell variable.
    echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    echo
fi
MAKE_OPTIONS=""
while [ "x${1}" != "x" ]; do
    case ${1} in
        fo*) # force
            shift
            echo Reconfigure build.
            FORCE_CMAKE="yes"
            ;;
        cm*) # cmake
            shift
            echo Only run CMAKE.  Do not compile.
            ONLY_CMAKE="yes"
            ;;
        clean) # clean
            shift
            echo Clean the build area
            RUN_CLEAN="yes"
            ;;
        ke*) # Keep going
            shift
            echo Continue on errors
            MAKE_OPTIONS=" -k ${MAKE_OPTIONS}"
            ;;
        te*|cte*) # test or ctest
            shift
            echo Run tests
            RUN_TEST="yes"
            ;;
        ve*) # verbose
            shift
            export VERBOSE=true
            ;;
        he*) # help
            echo
            echo "gundam-build [help|force|...] [-D<cmake-definitions>]"
            echo "   force -- Force cmake to ignore the cache"
            echo "   cmake -- Don't build the package (only run cmake)"
            echo "   clean -- Run make clean after cmake"
            echo "   keep  -- Keep going on a compilation error (make -k)"
            echo "   test <ctest-arguments> -- Run ctest on the build"
            echo "   verbose -- Run make with verbose turned on."
            echo "   help  -- This message"
            echo
            exit 0
            ;;
        -D*) # Add definitions and force cmake
            echo Add $1
            DEFINES="${DEFINES} ${1}"
            FORCE_CMAKE="yes"
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [ ! -f ${BUILD_LOCATION}/CMakeCache.txt -o ${FORCE_CMAKE} = "yes" ]; then
    echo cmake -B ${BUILD_LOCATION} -S ${GUNDAM_ROOT} --fresh ${DEFINES}
    cmake -B ${BUILD_LOCATION} -S ${GUNDAM_ROOT} --fresh ${DEFINES}
fi

if [ ${RUN_CLEAN} = "yes" ]; then
    echo cmake --build ${BUILD_LOCATION} --target clean
    cmake --build ${BUILD_LOCATION} --target clean
    echo Source cleaned.
    exit 0
fi

if [ ${ONLY_CMAKE} = "yes" ]; then
    exit 0
fi

echo cmake --build ${BUILD_LOCATION} \
     --parallel ${GUNDAM_JOBS} \
     -- ${MAKE_OPTIONS}
cmake --build ${BUILD_LOCATION} \
      --parallel ${GUNDAM_JOBS} \
      -- ${MAKE_OPTIONS} \
    || exit 1

echo cmake --install ${BUILD_LOCATION} --parallel ${GUNDAM_JOBS} || exit 1
cmake --install ${BUILD_LOCATION} --parallel ${GUNDAM_JOBS} || exit 1

if [ ${RUN_TEST} = "yes" ]; then
    echo ctest --test-dir ${BUILD_LOCATION} \
          --parallel ${GUNDAM_JOBS} \
          --output-on-failure \
          $*
    ctest --test-dir ${BUILD_LOCATION} \
          --parallel ${GUNDAM_JOBS} \
          --output-on-failure \
          $* \
        || exit 1
fi

echo "build:       " ${BUILD_LOCATION}
echo "installation:" ${GUNDAM_INSTALL}

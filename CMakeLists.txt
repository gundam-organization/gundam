cmake_minimum_required (VERSION 3.12 FATAL_ERROR)

###############################
# CMake project definition
###############################
project( GUNDAM )


###############################
# CMake options
###############################
option( WITH_GUNDAM_ROOT_APP "Build app gundamRoot" ON )
option( WITH_CACHE_MANAGER "Enable compiling of the cache manager used by the GPU" ON )
option( WITH_BREAKDOWN_CACHE "Use the breakdown cache" ON )
option( WITH_OPENMP "(OLD/xsllhFitter) Build with OpenMP libraries" OFF )
option( ENABLE_CUDA "Enable CUDA language check (Cache::Manager requires a GPU if CUDA is found)" OFF )

option( BUILD_DOC "Build documentation" OFF )
option( ENABLE_DEV_MODE "Enable specific dev related printouts" OFF  )
option( USE_STATIC_LINKS "Library build in static mod" OFF )
option( YAMLCPP_DIR "Set custom path to yaml-cpp lib" OFF )

option( COLOR_OUTPUT "Enable colored terminal output." ON )
option( TTYCHECK "Enable check if output is being sent to terminal/TTY." ON )
option( BATCH_MODE "Build to run in a batch queue (affects output)" OFF )

option( CXX_WARNINGS "Enable most C++ warning flags." ON )
option( CXX_MARCH_FLAG "Enable cpu architecture specific optimzations." OFF )
option( CMAKE_CXX_EXTENSIONS "Enable GNU extensions to C++ langauge (-std=gnu++14)." OFF )


###############################
# Cmake includes
###############################
include( CTest )
include( ${CMAKE_SOURCE_DIR}/cmake/cmessage.cmake )


# Printout version & generate header
cmessage( WARNING "Pre-checking GUNDAM version" )
set(GENERATE_DIR_FOR_VERSION_CHECK ${CMAKE_BINARY_DIR} CACHE STRING "GENERATE_DIR_FOR_VERSION_CHECK")
add_custom_target( preBuildVersionCheck
    COMMAND ${CMAKE_COMMAND} -D GENERATE_DIR_FOR_VERSION_CHECK=${GENERATE_DIR_FOR_VERSION_CHECK} -P ${CMAKE_SOURCE_DIR}/cmake/updateVersion.cmake
    COMMENT "Checking GUNDAM version with git files..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

# Run now as well
include( ${CMAKE_SOURCE_DIR}/cmake/updateVersion.cmake )

set( VERBOSE TRUE )


# Changes default install path to be a subdirectory of the build dir.
# Should set the installation dir at configure time with
# -DCMAKE_INSTALL_PREFIX=/install/path
if(NOT DEFINED CMAKE_INSTALL_PREFIX
    OR CMAKE_INSTALL_PREFIX STREQUAL ""
    OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
endif()

cmessage( STATUS "CMAKE_INSTALL_PREFIX: \"${CMAKE_INSTALL_PREFIX}\"")

# Set the default built type if it isn't already defined
if(NOT DEFINED CMAKE_BUILD_TYPE
    OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(BATCH_MODE)
    set(COLOR_OUTPUT NO)
    set(TTYCHECK NO)
    add_definitions( -D GUNDAM_BATCH )
endif(BATCH_MODE)

################################################################################
#                            Check Dependencies
################################################################################

#If you want to try an use the terminally buggy ROOT CMake scripts
# ROOTConfig.cmake -> usually in /install/dir/of/root/6.26.06_2/share/root/cmake

cmessage( WARNING "Looking for ROOT install..." )
find_package(
    ROOT
    REQUIRED COMPONENTS
    Geom Physics Matrix MathCore Tree RIO
    OPTIONAL_COMPONENTS
    Minuit2
)

if(ROOT_FOUND)
    cmessage(STATUS "[ROOT]: ROOT found")
    include(${ROOT_USE_FILE})
    # cmessage(STATUS "[ROOT]: ROOT packages found ${ROOT_LIBRARIES}")
    cmessage( STATUS "[ROOT]: ROOT include directory: ${ROOT_INCLUDE_DIRS}")
    cmessage(STATUS "[ROOT]: ROOT C++ Flags: ${ROOT_CXX_FLAGS}")

    if (NOT ROOT_minuit2_FOUND)
        # Minuit2 wasn't found, but make really sure before giving up.
        execute_process (COMMAND root-config --has-minuit2
            OUTPUT_VARIABLE ROOT_minuit2_FOUND
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif(NOT ROOT_minuit2_FOUND)

    # inc dir is $ROOTSYS/include/root
    set(CMAKE_ROOTSYS ${ROOT_INCLUDE_DIRS}/..)

else(ROOT_FOUND)
    cmessage( STATUS "find_package didn't find ROOT. Using shell instead...")

    # ROOT
    if(NOT DEFINED ENV{ROOTSYS} )
        cmessage(FATAL_ERROR "$ROOTSYS is not defined, please set up root first.")
    else()
        cmessage( STATUS "Using ROOT installed at $ENV{ROOTSYS}")
        set(CMAKE_ROOTSYS $ENV{ROOTSYS})
    endif()

    cmessage( STATUS "Including local GENERATE_ROOT_DICTIONARY implementation.")
    include(${CMAKE_SOURCE_DIR}/cmake/GenROOTDictionary.cmake)
    execute_process(COMMAND root-config --cflags
     OUTPUT_VARIABLE ROOT_CXX_FLAGS
     OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND root-config --libs
     OUTPUT_VARIABLE ROOT_LIBRARIES
     OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND root-config --version
     OUTPUT_VARIABLE ROOT_VERSION
     OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process (COMMAND root-config --ldflags
     OUTPUT_VARIABLE ROOT_LINK_FLAGS
     OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process (COMMAND root-config --has-minuit2
     OUTPUT_VARIABLE ROOT_minuit2_FOUND
     OUTPUT_STRIP_TRAILING_WHITESPACE)

    cmessage( STATUS "[ROOT]: root-config --version: ${ROOT_VERSION}")
    cmessage( STATUS "[ROOT]: root-config --libs: ${ROOT_LIBRARIES}")
    cmessage( STATUS "[ROOT]: root-config --cflags: ${ROOT_CXX_FLAGS}")
    cmessage( STATUS "[ROOT]: root-config --ldflags: ${ROOT_LINK_FLAGS}")

    add_compile_options("SHELL:${ROOT_CXX_FLAGS}")
    add_link_options("SHELL:${ROOT_LINK_FLAGS}")

endif(ROOT_FOUND)

# Try to figure out which version of C++ was used to compile ROOT.  ROOT
# generates header files that depend on the compiler version so we will
# need to use the same version.
execute_process(COMMAND root-config --has-cxx14 COMMAND grep yes
  OUTPUT_VARIABLE ROOT_cxx14_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND root-config --has-cxx17 COMMAND grep yes
  OUTPUT_VARIABLE ROOT_cxx17_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND root-config --has-cxx20 COMMAND grep yes
  OUTPUT_VARIABLE ROOT_cxx20_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if (NOT ROOT_minuit2_FOUND)
    cmessage( STATUS "[ROOT]:Rebuild root using -Dminuit2=on in the cmake command")
    cmessage(FATAL_ERROR "[ROOT]: minuit2 is required")
endif(NOT ROOT_minuit2_FOUND)


# NLOHMANN JSON
find_package(nlohmann_json)

if (nlohmann_json_FOUND)
    cmessage( STATUS "nlohmann JSON library found")
    # Additional actions for when the library is found
else()
    # sometimes the header can be found in ROOT...
    find_path(NLOHMANN_JSON_INCLUDE_DIR NAMES nlohmann/json.hpp)

    if (NLOHMANN_JSON_INCLUDE_DIR)
        cmessage( STATUS "nlohmann JSON header found: ${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
        # Additional actions for when the library is found
    else()
        cmessage( FATAL_ERROR "nlohmann JSON library not found")
        # Additional actions for when the library is not found
    endif()
endif()


# YAML-CPP
cmessage( WARNING "Looking for YAML install..." )
find_package( yaml-cpp REQUIRED HINTS ${YAMLCPP_DIR} )
if(NOT yaml-cpp_FOUND)
    cmessage(FATAL_ERROR "yaml-cpp library not found.")
endif()
include_directories( ${YAML_CPP_INCLUDE_DIR} )
cmessage( STATUS "Custom yaml-cpp installation: ${YAMLCPP_DIR}")
cmessage( STATUS "yaml-cpp include directory: ${YAML_CPP_INCLUDE_DIR}")
cmessage( STATUS "yaml-cpp lib: ${YAML_CPP_LIBRARIES}")
if( "${YAML_CPP_INCLUDE_DIR} " STREQUAL " ")
    # WORKAROUND FOR CCLYON (old cmake version/pkg)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    set(YAMLCPP_INSTALL_DIR ${YAMLCPP_DIR})
    find_package( YAMLCPP REQUIRED )
    if( NOT YAMLCPP_FOUND )
        cmessage(FATAL_ERROR "yaml-cpp library not found.")
    endif()
    cmessage( STATUS " - yaml-cpp include directory: ${YAMLCPP_INCLUDE_DIR}")
    cmessage( STATUS " - yaml-cpp lib: ${YAMLCPP_LIBRARY}")
    if( "${YAMLCPP_INCLUDE_DIR} " STREQUAL " ")
        cmessage(FATAL_ERROR "empty YAMLCPP_INCLUDE_DIR returned.")
    endif()
    include_directories( ${YAMLCPP_INCLUDE_DIR} )
    set(YAML_CPP_LIBRARIES ${YAMLCPP_LIBRARY})
    #    list(APPEND LINK_LIBARY_LIST ${YAMLCPP_LIBRARY})
endif()


cmessage( WARNING "Looking for optional ZLib install..." )
find_package(ZLIB)
if (${ZLIB_FOUND})
    cmessage( STATUS "ZLIB found : ${ZLIB_VERSION_STRING}")
    cmessage( STATUS "ZLIB_INCLUDE_DIRS = ${ZLIB_INCLUDE_DIRS}")
    cmessage( STATUS "ZLIB_LIBRARIES = ${ZLIB_LIBRARIES}")
    add_definitions( -D USE_ZLIB=1 )
else()
    add_definitions( -D USE_ZLIB=0 )
endif ()


# Check for the availability of CUDA
if( ENABLE_CUDA )
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        cmessage( STATUS "CUDA support enabled")
        enable_language(CUDA)
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # The default is taken from the CUDAARCHS environment
            # variable.  If it isn't set, then set it to the earliest
            # non-deprecated architecture.
            #   2022: architectures before 52 are deprecated.
            if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.23)
                # After cmake 3.23, this can be set to all or all-major
                set(CMAKE_CUDA_ARCHITECTURES all)
            else()
                set(CMAKE_CUDA_ARCHITECTURES 52)
            endif()
        endif()
        cmessage( STATUS "CUDA compilation architectures: \"${CMAKE_CUDA_ARCHITECTURES}\"")
        cmessage(WARNING "The \"--cache-manager\" option requires a GPU")
    else(CMAKE_CUDA_COMPILER)
        cmessage(WARNING "CUDA not present -- Cache::Manager use the CPU")
    endif(CMAKE_CUDA_COMPILER)
else( ENABLE_CUDA )
    cmessage( ALERT "CUDA support disabled" )
endif( ENABLE_CUDA )


if (WITH_CACHE_MANAGER)
    cmessage( WARNING "Enabling cache manager..." )
    add_definitions( -D GUNDAM_USING_CACHE_MANAGER )

    # uncomment to enable the slow validations (NEVER during productions
    # or normal running).  These are whole code validations and are
    # extremely slow.
    if (CACHE_MANAGER_SLOW_VALIDATION)
        cmessage( STATUS "Using slow validation for debugging" )
        cmessage(WARNING "Using slow validation so runs will be very slow" )
        add_definitions( -DCACHE_MANAGER_SLOW_VALIDATION)
    endif (CACHE_MANAGER_SLOW_VALIDATION)

    cmessage( STATUS "GPU support is enabled (compiled, but only used when CUDA enabled)" )
else()
    cmessage( ALERT "GPU support is disabled" )
endif()

set(WITH_GENERIC_SPLINES ON)
if(NOT WITH_GENERIC_SPLINES)
    if(WITH_CACHE_MANAGER)
        cmessage(WARNING "Cache::Manager works best with generic splines" )
    else(WITH_CACHE_MANAGER)
        add_definitions(-D USE_TSPLINE3_EVAL)
    endif(WITH_CACHE_MANAGER)
endif(NOT WITH_GENERIC_SPLINES)

if(WITH_BREAKDOWN_CACHE)
    add_definitions(-D USE_BREAKDOWN_CACHE)
else()
    cmessage(WARNING "Not using breakdown cache" )
endif(WITH_BREAKDOWN_CACHE)

################################################################################
#                       SubModules
################################################################################

# SubModules: These are just adding the code directly, and not being compiled
# as stand-alone projects.  This has the danger that it's not fully configured,
# but should be OK

include( ${CMAKE_SOURCE_DIR}/cmake/checkSubmodules.cmake )

checkSubmodule( cpp-generic-toolbox )
checkSubmodule( simple-cpp-logger )
checkSubmodule( simple-cpp-cmd-line-parser )


## Add the CmdLineParser
# Reproduce needed parts of the simple-cpp-cmd-line-parser CMakeLists.txt
include_directories(submodules/simple-cpp-cmd-line-parser/include)
if(yaml-cpp_FOUND)
    add_definitions( -DCMDLINEPARSER_YAML_CPP_ENABLED=1 )
endif()

## Add the GenericToolbox²
# Reproduce needed parts of the cpp-generic-toolbox CMakeLists.txt
include_directories(submodules/cpp-generic-toolbox/include)

#file( GLOB CPP_GENERIC_TOOLBOX_HEADERS ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/*.h )
#file( GLOB CPP_GENERIC_TOOLBOX_HEADERS_IMPL ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/implementation/*.h )
#install(FILES ${CPP_GENERIC_TOOLBOX_HEADERS} DESTINATION include)
#install(FILES ${CPP_GENERIC_TOOLBOX_HEADERS_IMPL} DESTINATION include/implementation)

add_definitions( -D PROGRESS_BAR_FILL_TAG="\\\"T2K\#"\\\" )
if (COLOR_OUTPUT)
    add_definitions( -D PROGRESS_BAR_ENABLE_RAINBOW=1 )
else (COLOR_OUTPUT)
    # add_definitions( -D PROGRESS_BAR_ENABLE_RAINBOW=0 )
    add_definitions( -D CPP_GENERIC_TOOLBOX_NOCOLOR )
endif (COLOR_OUTPUT)
if (BATCH_MODE)
    add_definitions( -D CPP_GENERIC_TOOLBOX_BATCH )
endif(BATCH_MODE)

## Add the Logger
# Reproduce needed parts of the simple-cpp-logger CMakeLists.txt
include_directories(submodules/simple-cpp-logger/include)
add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
add_definitions( -D LOGGER_PREFIX_LEVEL=3 )
add_definitions( -D LOGGER_TIME_FORMAT="\\\"%d/%m/%Y %H:%M:%S"\\\" )

if(${CMAKE_BUILD_TYPE} MATCHES Debug OR ${ENABLE_DEV_MODE})
    cmessage( STATUS "Logger set in dev mode." )
    add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER} {FILELINE}"\\\" )
else()
    cmessage( STATUS "Logger set in release mode." )
    add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER}"\\\" )
endif()

if(NOT COLOR_OUTPUT)
    cmessage( STATUS "Color output is disabled." )
    add_definitions( -D LOGGER_ENABLE_COLORS=0 )
    add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=0 )
else()
    add_definitions( -D LOGGER_ENABLE_COLORS=1 )
    add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=1 )
endif()


################################################################################
#                            Compiler Setup
################################################################################

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.5)
        cmessage(STATUS "Detected GCC version: ${CMAKE_CXX_COMPILER_VERSION}")
        cmessage(FATAL_ERROR "GCC version must be at least 5.0")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" )
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
        cmessage( STATUS "Detected Clang version: ${CMAKE_CXX_COMPILER_VERSION}" )
        cmessage(FATAL_ERROR "Clang version must be at least 3.3" )
    endif()
else()
    cmessage(WARNING "You are using an untested compiler." )
endif()

# CXX standard is required and must match the version ROOT was compiled with.  
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set the compiler version so that it will match the
# compiler that was used to compile ROOT.  Recent ROOT documentation
# explicitly notes that the appliation needs to use the same C++
# standard as ROOT.
if ( ROOT_cxx14_FOUND )
  cmessage(STATUS "ROOT compiled with C++14")
  set(CMAKE_CXX_STANDARD 14)
elseif ( ROOT_cxx17_FOUND )
  cmessage(STATUS "ROOT compiled with C++17")
  set(CMAKE_CXX_STANDARD 17)
elseif ( ROOT_cxx20_FOUND )
  cmessage(STATUS "ROOT compiled with C++20")
  set(CMAKE_CXX_STANDARD 20)
else ( ROOT_cxx14_FOUND )
  cmessage(WARNING "ROOT C++ standard not set, use ROOT minimum (C++14)")
  set(CMAKE_CXX_STANDARD 14)
endif ( ROOT_cxx14_FOUND)

if( WITH_OPENMP )
    cmessage( STATUS "Compiling with OpenMP" )
    include(FindOpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
        cmessage( STATUS "OpenMP include directory: ${OpenMP_CXX_INCLUDE_DIR}" )
        cmessage( STATUS "OpenMP libs: ${OpenMP_CXX_LIB_NAMES}" )
    else()
        cmessage(ERROR "OpenMP not found." )
    endif()
else()
    #    cmessage( STATUS "Compiling without OpenMP, enable with \"-D WITH_OPENMP\"" )
endif()

if(CXX_MARCH_FLAG)
endif()

if(CXX_WARNINGS)
    cmessage( STATUS "Enable CXX warnings" )
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>)
    # add_compile_options(-Wformat=0)
    # add_compile_options(-Wno-reorder)
endif()

if(NOT COLOR_OUTPUT)
    add_definitions(-D NOCOLOR)
endif()

if(TTYCHECK)
    add_definitions(-D TTYCHECK)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -g")

################################################################################
# CMake Generated
###############################################################################

cmessage (STATUS "C++ Compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
cmessage (STATUS "C++ Standard      : ${CMAKE_CXX_STANDARD}")
cmessage (STATUS "C++ Release flags : ${CMAKE_CXX_FLAGS_RELEASE}")
cmessage (STATUS "C++ Debug flags   : ${CMAKE_CXX_FLAGS_DEBUG}")
cmessage (STATUS "Build type        : ${CMAKE_BUILD_TYPE}")

################################################################################
# Doxygen
###############################################################################

if( BUILD_DOC )
    # check if Doxygen is installed
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        # set input and output files
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxygen.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started" )

        # note the option ALL which allows to build the docs together with the application
        add_custom_target( doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )
    else (DOXYGEN_FOUND)
        message("Doxygen need to be installed to generate the doxygen documentation" )
    endif (DOXYGEN_FOUND)
endif()


################################################################################
#                            Specify Target Subdirs
################################################################################

set( MODULES Utils)

# Add the basic modules
list( APPEND MODULES DialDictionary)
list( APPEND MODULES ParametersManager)
list( APPEND MODULES SamplesManager)
list( APPEND MODULES DatasetManager)
list( APPEND MODULES Propagator)
list( APPEND MODULES Fitter)

if(WITH_CACHE_MANAGER)
    list( APPEND MODULES CacheManager )
endif()

cmessage( WARNING "Configuring modules..." )
foreach(mod ${MODULES})
    cmessage( STATUS "Adding cmake module: ${mod}" )
    add_subdirectory( ${CMAKE_SOURCE_DIR}/src/${mod} )
endforeach()

cmessage( WARNING "Configuring applications..." )
add_subdirectory( ${CMAKE_SOURCE_DIR}/src/Applications )

add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )

configure_file(cmake/build_setup.sh.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh"
    DESTINATION ${CMAKE_INSTALL_PREFIX})

cmessage( STATUS "Identified Version ${GUNDAM_VERSION_STRING}" )

# A command that works to compile gundam from any sub-directory.  You
# can get this aliased to "gundam-build" by sourcing
#
# . "$(git rev-parse --show-toplevel)/cmake/gundam-setup.sh"
#
# That runs:
# Local Variables:
# compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
# End:

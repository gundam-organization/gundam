message("")
cmessage( STATUS "Checking dependencies...")

include( FetchContent )
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)

# A string with all the packages to be made available after everything
# is declared
set(DeclaredContent "")

##########
# ROOT
##########

#If you want to try an use the terminally buggy ROOT CMake scripts
# ROOTConfig.cmake -> usually in /install/dir/of/root/6.26.06_2/share/root/cmake

cmessage( STATUS "Looking for ROOT install..." )
find_package(
    ROOT
    COMPONENTS
    Tree Minuit2 Matrix
    Physics MathCore RIO
)

if( ROOT_FOUND )
  if (NOT ROOT_config_CMD)
    set(ROOT_config_CMD ${ROOT_root_CMD}-config)
  endif(NOT ROOT_config_CMD)

  cmessage( STATUS "[ROOT]: ROOT found." )
  cmessage( STATUS "[ROOT]: ROOT cmake use file ${ROOT_USE_FILE}")
  cmessage( STATUS "[ROOT]: ROOT include directory: ${ROOT_INCLUDE_DIRS}" )
  cmessage( STATUS "[ROOT]: ROOT C++ Flags: ${ROOT_CXX_FLAGS}" )

  execute_process(COMMAND ${ROOT_config_CMD} --version
      OUTPUT_VARIABLE ROOT_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  cmessage( STATUS "[ROOT]: ROOT Version: ${ROOT_VERSION}" )

  # Grab functions such as generate dictionary
  include( ${ROOT_USE_FILE} )

  # Add the libraries and include files here.  These "should" go in
  # the target_link_libraries or target_include_directories for each library
  # that uses them, but they all use them, so put them here. This keeps the
  # symmetry when root-config (if find_package didn't work)
  link_libraries(${ROOT_LIBRARIES})
  include_directories(${ROOT_INCLUDE_DIRS})

  if (ROOT_VERSION VERSION_GREATER_EQUAL 6.30.00)
    set(ROOT_minuit2_FOUND "yes")
  endif()

else( ROOT_FOUND )
  cmessage( WARNING "find_package didn't find ROOT. Using shell instead...")

  # ROOT
  if(NOT DEFINED ENV{ROOTSYS} )
    cmessage( FATAL_ERROR "$ROOTSYS is not defined, please set up root first." )
  else()
    cmessage( STATUS "Using ROOT installed at $ENV{ROOTSYS}")
  endif()

  if (NOT ROOT_config_CMD)
    set(ROOT_config_CMD root-config)
  endif(NOT ROOT_config_CMD)

  cmessage( STATUS "Including local GENERATE_ROOT_DICTIONARY implementation." )
  include(${CMAKE_SOURCE_DIR}/cmake/utils/GenROOTDictionary.cmake)
  execute_process(COMMAND ${ROOT_config_CMD} --cflags
      OUTPUT_VARIABLE ROOT_CXX_FLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${ROOT_config_CMD} --libs
      OUTPUT_VARIABLE ROOT_LIBRARIES
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${ROOT_config_CMD} --version
    OUTPUT_VARIABLE ROOT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process (COMMAND ${ROOT_config_CMD} --ldflags
    OUTPUT_VARIABLE ROOT_LINK_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process (COMMAND ${ROOT_config_CMD} --has-minuit2
    OUTPUT_VARIABLE ROOT_minuit2_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  cmessage( STATUS "[ROOT]: root-config --version: ${ROOT_VERSION}")
  cmessage( STATUS "[ROOT]: root-config --libs: ${ROOT_LIBRARIES}")
  cmessage( STATUS "[ROOT]: root-config --cflags: ${ROOT_CXX_FLAGS}")
  cmessage( STATUS "[ROOT]: root-config --ldflags: ${ROOT_LINK_FLAGS}")

  add_compile_options("SHELL:${ROOT_CXX_FLAGS}")
  add_link_options("SHELL:${ROOT_LINK_FLAGS}")

  if (ROOT_VERSION VERSION_GREATER_EQUAL 6.30.00)
    set(ROOT_minuit2_FOUND "yes")
  endif()

endif( ROOT_FOUND )

# Try to figure out which version of C++ was used to compile ROOT.  ROOT
# generates header files that depend on the compiler version so we will
# need to use the same version.
execute_process(COMMAND ${ROOT_config_CMD} --has-cxx14 COMMAND grep yes
  OUTPUT_VARIABLE ROOT_cxx14_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${ROOT_config_CMD} --has-cxx17 COMMAND grep yes
  OUTPUT_VARIABLE ROOT_cxx17_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${ROOT_config_CMD} --has-cxx20 COMMAND grep yes
  OUTPUT_VARIABLE ROOT_cxx20_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Extract the home location for ROOT.  This is the value of ROOTSYS
execute_process (COMMAND ${ROOT_config_CMD} --prefix
  OUTPUT_VARIABLE CMAKE_ROOTSYS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Minuit2 wasn't found, but make really sure before giving up.
if (NOT ROOT_minuit2_FOUND)
  execute_process (COMMAND ${ROOT_config_CMD} --has-minuit2
    OUTPUT_VARIABLE ROOT_minuit2_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif(NOT ROOT_minuit2_FOUND)

# If we truly don't have minuit2, then complain
if (NOT ROOT_minuit2_FOUND AND NOT WITH_MINUIT2_MISSING)
  cmessage(WARNING "[ROOT]: Use >6.30 or rebuild root with -Dminuit2=on")
  cmessage(WARNING "[ROOT]: Set WITH_MINUIT2_MISSING OFF to disable check")
  cmessage(FATAL_ERROR "[ROOT]: minuit2 is required")
endif()


####################
# NLOHMANN JSON
####################

cmessage( STATUS "Looking for JSON install..." )
find_package(nlohmann_json 3.11.3 EXACT CONFIG)
if( NOT nlohmann_json_FOUND )
  cmessage( WARNING "System nlohmann_json package not found")
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW true
    # Only do the basic compile
    CMAKE_ARGS -DJSON_BuildTests=OFF -DJSON_CI=OFF
    # OVERRIDE_FIND_PACKAGE
  )
  set(DeclaredContent ${DeclaredContent} nlohmann_json)
endif( NOT nlohmann_json_FOUND )


####################
# YAML-CPP
####################
cmessage( STATUS "Looking for YAML install..." )

find_package(yaml-cpp 0.9.0 EXACT CONFIG)
if( NOT yaml-cpp_FOUND )
  cmessage( WARNING "System yaml-cpp package not found")
  FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.9.0
    # Make sure yaml-cpp doesn't mess with gtest
    CMAKE_ARGS -DGTEST_INSTALL=OFF
    # OVERRIDE_FIND_PACKAGE
  )
  set(DeclaredContent ${DeclaredContent} yaml-cpp)
endif()


####################
# GoogleTest
####################
find_package(GTest 1.16.0 EXACT CONFIG)
if( NOT GTest_FOUND )
  cmessage( WARNING "System GTest package not found")
  FetchContent_Declare(
    GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
    # OVERRIDE_FIND_PACKAGE
  )
  set(DeclaredContent ${DeclaredContent} GTest)
endif( NOT GTest_FOUND )

####################
# CUDA (optional)
####################

# Check for the availability of CUDA
if( WITH_CUDA_LIB )
  cmessage( STATUS "WITH_CUDA_LIB=ON: Checking for CUDA support...")
  cmake_minimum_required( VERSION 3.12 FATAL_ERROR )
  include( CheckLanguage )
  check_language( CUDA )
  if( CMAKE_CUDA_COMPILER )
    cmessage( STATUS "CUDA support enabled." )
    enable_language(CUDA)
    if( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
      # The default is taken from the CUDAARCHS environment
      # variable.  If it isn't set, then set it to the earliest
      # non-deprecated architecture.
      #   2022: architectures before 52 are deprecated.
      if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.23 )
        # After cmake 3.23, this can be set to all or all-major
        set( CMAKE_CUDA_ARCHITECTURES all )
      else()
        cmessage( WARNING "Force CUDA architecture to 52 (deprecated)")
        set( CMAKE_CUDA_ARCHITECTURES 52 )
      endif()
    endif()
    cmessage( STATUS "CUDA compilation architectures: \"${CMAKE_CUDA_ARCHITECTURES}\"")
    cmessage( ALERT "Running with \"--gpu\" option will require a GPU" )
  else( CMAKE_CUDA_COMPILER )
    cmessage( FATAL_ERROR "Option WITH_CUDA_LIB=ON: CUDA not present." )
  endif( CMAKE_CUDA_COMPILER )
else( WITH_CUDA_LIB )
  if( WITH_CACHE_MANAGER )
    cmessage( STATUS "WITH_CACHE_MANAGER=ON: CUDA support disabled. Use -D WITH_CUDA_LIB=ON if needed." )
  endif()
endif( WITH_CUDA_LIB )

####################
# FetchContent packages.
####################

if (DeclaredContent)
  # Make any FetchContent available.  Fetched packages should be added
  # to the local DeclaredContent variable.
  cmessage(WARNING "FetchContent: Will build ${DeclaredContent}")
  FetchContent_MakeAvailable(${DeclaredContent})
else()
  cmessage(WARNING "No content declared")
endif (DeclaredContent)

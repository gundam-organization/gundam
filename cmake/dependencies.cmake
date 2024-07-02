message("")
cmessage( WARNING "Checking dependencies...")


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

else( ROOT_FOUND )
  cmessage( STATUS "find_package didn't find ROOT. Using shell instead...")

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

execute_process (COMMAND ${ROOT_config_CMD} --prefix
  OUTPUT_VARIABLE CMAKE_ROOTSYS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Minuit2 wasn't found, but make really sure before giving up.
if (NOT ROOT_minuit2_FOUND)
  execute_process (COMMAND ${ROOT_config_CMD} --has-minuit2
    OUTPUT_VARIABLE ROOT_minuit2_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif(NOT ROOT_minuit2_FOUND)

if (NOT ROOT_minuit2_FOUND)
  cmessage( STATUS "[ROOT]: Use >6.32 or rebuild root with-Dminuit2=on in the cmake command")
  cmessage(FATAL_ERROR "[ROOT]: minuit2 is required")
endif(NOT ROOT_minuit2_FOUND)

include_directories( ${ROOT_INCLUDE_DIR} )


####################
# NLOHMANN JSON
####################

cmessage( STATUS "Looking for JSON install..." )
find_package( nlohmann_json QUIET )

if( nlohmann_json_FOUND )
  cmessage( STATUS "nlohmann JSON library found.")
  link_libraries( nlohmann_json::nlohmann_json )
else()
  cmessage( ALERT "nlohmann JSON library not found. Using fetch content... (CMake version >= 3.11)")
  cmake_minimum_required( VERSION 3.11 FATAL_ERROR )
  include( FetchContent )

  FetchContent_Declare( json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz )
  FetchContent_MakeAvailable( json )

  include_directories( ${nlohmann_json_SOURCE_DIR}/include )
  cmessage( STATUS "nlohmann JSON library fetched: ${nlohmann_json_SOURCE_DIR}/include")
endif()


####################
# YAML-CPP
####################

cmessage( STATUS "Looking for YAML install..." )

# WORKAROUND FOR CCLYON (old cmake version/pkg)
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils )
set( CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils )

if( DEFINED $YAMLCPP_DIR )
  cmessage( ALERT "Setting yaml-cpp hint to ${YAMLCPP_DIR}." )
  set( YAMLCPP_INSTALL_DIR ${YAMLCPP_DIR} )
endif()

find_package( YAMLCPP REQUIRED HINTS ${YAMLCPP_DIR} )
if( NOT YAMLCPP_FOUND )
  cmessage(FATAL_ERROR "yaml-cpp library not found.")
endif()
cmessage( STATUS " - yaml-cpp include directory: ${YAMLCPP_INCLUDE_DIR}")
cmessage( STATUS " - yaml-cpp lib: ${YAMLCPP_LIBRARY}")
if( "${YAMLCPP_INCLUDE_DIR} " STREQUAL " ")
  cmessage(FATAL_ERROR "empty YAMLCPP_INCLUDE_DIR returned.")
endif()
set(YAML_CPP_LIBRARIES ${YAMLCPP_LIBRARY})
include_directories( ${YAMLCPP_INCLUDE_DIR} )
link_libraries( ${YAML_CPP_LIBRARIES} )


####################
# ZLIB (optional)
####################

if( ${DISABLE_ZLIB} )
  cmessage( WARNING "DISABLE_ZLIB=ON. Not using Zlib." )
  add_definitions( -D USE_ZLIB=0 )
else()
  cmessage( STATUS "Looking for optional ZLib install..." )
  find_package(ZLIB)
  if (${ZLIB_FOUND})
    cmessage( STATUS "ZLIB found : ${ZLIB_VERSION_STRING}")
    cmessage( STATUS "ZLIB_INCLUDE_DIRS = ${ZLIB_INCLUDE_DIRS}")
    cmessage( STATUS "ZLIB_LIBRARIES = ${ZLIB_LIBRARIES}")

    add_definitions( -D USE_ZLIB=1 )
    include_directories( ${ZLIB_INCLUDE_DIRS} )
    link_libraries( ${ZLIB_LIBRARIES} )
  else()
    cmessage( WARNING "ZLib not found. Will compile without the associated features." )
    add_definitions( -D USE_ZLIB=0 )
  endif ()
endif ()


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
        set( CMAKE_CUDA_ARCHITECTURES 52 )
      endif()
    endif()
    cmessage( STATUS "CUDA compilation architectures: \"${CMAKE_CUDA_ARCHITECTURES}\"")
    cmessage( ALERT "Running with \"--cache-manager\" option requires a GPU" )
  else( CMAKE_CUDA_COMPILER )
    cmessage( FATAL_ERROR "Option WITH_CUDA_LIB=ON: CUDA not present." )
  endif( CMAKE_CUDA_COMPILER )
else( WITH_CUDA_LIB )
  if( WITH_CACHE_MANAGER )
    cmessage( STATUS "WITH_CACHE_MANAGER=ON: CUDA support disabled. Use -D WITH_CUDA_LIB=ON if needed." )
  endif()
endif( WITH_CUDA_LIB )

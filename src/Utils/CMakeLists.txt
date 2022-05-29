set(SRCFILES
        src/GlobalVariables.cc
        src/DataBinSet.cpp
        src/DataBin.cpp
        src/JsonUtils.cpp
        src/YamlUtils.cpp
        src/GundamGreetings.cpp
        src/ScanConfig.cpp
        )
set(HEADERS
        include/GlobalVariables.h
        include/DataBin.h
        include/Likelihoods.hh
        include/DataBinSet.h
        include/JsonUtils.h
        include/YamlUtils.h
        include/YamlUtils.impl.h
        include/GundamGreetings.h
        include/ScanConfig.h
        include/CalculateMonotonicSpline.h
        include/CalculateUniformSpline.h
        include/CalculateGeneralSpline.h
        )

if( USE_STATIC_LINKS )
  add_library(GundamUtils STATIC ${SRCFILES})
else()
  add_library(GundamUtils SHARED ${SRCFILES})
endif()

# Make sure the current directories are available for the later
# compilation.
target_include_directories( GundamUtils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( GundamUtils PUBLIC
        ${ROOT_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        )

#set_target_properties(GundamUtils PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS GundamUtils DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)


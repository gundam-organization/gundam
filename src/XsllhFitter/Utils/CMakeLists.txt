set(SRCFILES
        src/BinManager.cc
        src/CalcChisq.cc
        src/EigenDecomp.cc
        src/OptParser.cc
        src/PlotStyle.cc
        src/ToyThrower.cc
        src/GeneralizedFitBin.cc
        )
set(HEADERS
        include/BinManager.hh
        include/CalcChisq.hh
        include/EigenDecomp.hh
        include/OptParser.hh
        include/PlotStyle.hh
        include/ToyThrower.hh
        include/GlobalVariables.h
        include/GeneralizedFitBin.h
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(XsllhFitterUtils STATIC ${SRCFILES})
else()
  add_library(XsllhFitterUtils SHARED ${SRCFILES})
endif()

# Make sure the current directories are available for the later compilation.
target_include_directories( XsllhFitterUtils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

target_link_libraries( XsllhFitterUtils PUBLIC ${ROOT_LIBRARIES} ${YAML_CPP_LIBRARIES} )

#set_target_properties(XsllhFitterUtils PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS XsllhFitterUtils DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)


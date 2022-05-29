set(SRCFILES
        src/AnaFitParameters.cpp
        src/DetParameters.cpp
        src/FitParameters.cc
        src/FluxParameters.cc
        src/XsecParameters.cc
        src/XsecDial.cc
        )

if( USE_STATIC_LINKS )
  add_library(XsllhFitterParameters STATIC ${SRCFILES})
else()
  add_library(XsllhFitterParameters SHARED ${SRCFILES})
endif()

target_include_directories(XsllhFitterParameters PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Dependencies
target_link_libraries( XsllhFitterParameters PUBLIC
        XsllhFitterAnaevents
        XsllhFitterUtils
        ${ROOT_LIBRARIES}
)

#set_target_properties(XsllhFitterParameters PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS XsllhFitterParameters DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)


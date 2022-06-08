set(SRCFILES
        src/FitParameterSet.cpp
        src/FitParameter.cpp
        src/DialSet.cpp
        src/Dial.cpp
        src/NormDial.cpp
        src/SplineDial.cpp
        src/GraphDial.cpp
        src/GraphDialLin.cpp
        src/NestedDialTest.cpp
        )

if( USE_STATIC_LINKS )
  add_library(GundamFitParameters STATIC ${SRCFILES})
else()
  add_library(GundamFitParameters SHARED ${SRCFILES})
endif()

target_include_directories(GundamFitParameters PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Dependencies
target_link_libraries( GundamFitParameters PUBLIC GundamUtils ${ROOT_LIBRARIES} )

#set_target_properties(GundamFitParameters PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS GundamFitParameters DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)


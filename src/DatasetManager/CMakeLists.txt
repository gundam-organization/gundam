set(SRCFILES
        src/DatasetLoader.cpp
        src/EventTreeWriter.cpp
        src/DataDispenser.cpp
    )

set(HEADERS
        include/DatasetLoader.h
        include/EventTreeWriter.h
        include/DataDispenser.h
)

if( USE_STATIC_LINKS )
  add_library(GundamDatasetManager STATIC ${SRCFILES})
else()
  add_library(GundamDatasetManager SHARED ${SRCFILES})
endif()

target_include_directories(GundamDatasetManager PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( GundamDatasetManager
        GundamFitParameters
        GundamFitSamples
        GundamDials
        ${ROOT_LIBRARIES}
        )

#set_target_properties(GundamDatasetManager PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS GundamDatasetManager DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)


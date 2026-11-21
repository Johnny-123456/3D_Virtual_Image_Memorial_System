cmake_minimum_required(VERSION 3.10)
project(memorial_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 5.9 REQUIRED COMPONENTS Widgets Gui OpenGL)
find_package(OpenGL REQUIRED)

add_executable(memorial_viewer
    main.cpp
    dibrwidget.cpp
    dibrwidget.h
)

target_link_libraries(memorial_viewer
    Qt5::Widgets
    Qt5::Gui
    Qt5::OpenGL
    OpenGL::GL
)

install(TARGETS memorial_viewer RUNTIME DESTINATION bin)

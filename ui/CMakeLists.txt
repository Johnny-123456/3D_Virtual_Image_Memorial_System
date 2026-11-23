cmake_minimum_required(VERSION 3.10)
project(memorial_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 5.9 REQUIRED COMPONENTS Widgets Gui OpenGL Network)
find_package(OpenGL REQUIRED)

# --- 最小查看器（开发/调试用，支持 --snapshot 离屏截图）---
add_executable(memorial_viewer
    main.cpp
    dibrwidget.cpp
    dibrwidget.h
    eyelink.h
)
target_link_libraries(memorial_viewer
    Qt5::Widgets Qt5::Gui Qt5::OpenGL Qt5::Network OpenGL::GL)

# --- 完整系统界面（首页 / 采集 / 观看）---
add_executable(memorial_studio
    studio.cpp
    dibrwidget.cpp
    dibrwidget.h
    eyelink.h
)
target_link_libraries(memorial_studio
    Qt5::Widgets Qt5::Gui Qt5::OpenGL Qt5::Network OpenGL::GL)

install(TARGETS memorial_viewer memorial_studio RUNTIME DESTINATION bin)

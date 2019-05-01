cmake_minimum_required(VERSION 3.13)
project(student)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(student
        application.c
        application.h
        bunny.c
        bunny.h
        callbackAndData.c
        callbackAndData.h
        camera.c
        camera.h
        cpu.c
        cpu.h
        drawPoints.cpp
        drawTriangles.cpp
        emptyMethod.c
        emptyMethod.h
        fwd.h
        globals.c
        globals.h
        gpu.c
        gpu.h
        linearAlgebra.c
        linearAlgebra.h
        main.c
        method.c
        method.h
        parseArguments.c
        parseArguments.h
        phongMethod.c
        phongMethod.h
        pointBoxMethod.c
        pointBoxMethod.h
        pointCircleMethod.c
        pointCircleMethod.h
        pointMethod.c
        pointMethod.h
        pointSquareMethod.c
        pointSquareMethod.h
        structures.h
        triangle3DMethod.c
        triangle3DMethod.h
        triangleMethod.c
        triangleMethod.h
        window.c
        window.h)

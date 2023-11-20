set(target_name "simulator")
add_library(${target_name} SHARED)
add_library("xuan::${target_name}" ALIAS ${target_name})
target_compile_features(${target_name}
    PUBLIC
    cxx_std_17
)
target_sources(${target_name}
    PUBLIC
    simulator.cu
)
target_include_directories(${target_name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(${target_name}
    PUBLIC
    -fPIC
)
target_link_libraries(${target_name}
    PUBLIC
    xuan::neighbor_list
    xuan::particles
)

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


set(TARGET_NAME ov_core_unit_tests)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

ov_try_use_gold_linker()

add_definitions(-DSERIALIZED_ZOO=\"${TEST_MODEL_ZOO}/core/models\")

message(STATUS "OpenVINO Core unit tests enabled")

# For type relaxed types
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/threading.cpp
        PROPERTIES INCLUDE_DIRECTORIES $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set_source_files_properties(ov_tensor_test.cpp
                                type_prop/multiclass_nms.cpp
                                PROPERTIES COMPILE_OPTIONS -Wno-suggest-override)
endif()

list(APPEND UNIT_TESTS_DEPENDENCIES openvino_template_extension)
list(APPEND UNIT_TESTS_DEPENDENCIES template_extension)

list(APPEND EXCLUDE_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/dnnl.cpp)

ov_add_test_target(
    NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        EXCLUDED_SOURCE_PATHS
            ${EXCLUDE_TESTS}
        DEPENDENCIES
            ${UNIT_TESTS_DEPENDENCIES}
            # process models
            test_model_zoo
        LINK_LIBRARIES
            common_test_utils
            openvino::reference
            openvino::builders
            openvino::util
            openvino::shape_inference
            ${CMAKE_DL_LIBS}
            Threads::Threads
            openvino::conditional_compilation
            openvino::runtime::dev
        ADD_CLANG_FORMAT
        LABELS
            OV
            IE
            CORE
)

get_target_property(OV_CORE_SRC_DIR ngraph_obj SOURCE_DIR)
target_include_directories(${TARGET_NAME} PRIVATE ${OV_CORE_SRC_DIR}/src
                                                  ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        FRONTEND_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}${FRONTEND_NAME_PREFIX}"
        # Assume <lib>.so is an existed symlink to <lib><version>.so (or <lib>.so<version>
        FRONTEND_LIB_SUFFIX="${FRONTEND_NAME_SUFFIX}${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )

add_dependencies(${TARGET_NAME} ${UNIT_TESTS_DEPENDENCIES})

if (ENABLE_OV_ONNX_FRONTEND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_OV_ONNX_FRONTEND)
endif()

if (OV_COMPILER_IS_CLANG)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

if(ENABLE_OV_IR_FRONTEND)
    add_dependencies(${TARGET_NAME} openvino_ir_frontend)
endif()

add_subdirectory(frontend)

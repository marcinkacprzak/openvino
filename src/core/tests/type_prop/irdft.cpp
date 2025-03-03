// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/irdft.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using namespace testing;

struct IRDFTConstantAxesAndConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    std::vector<label_t> expected_labels;
};

struct IRDFTConstantAxesAndConstantSignalSizeTest
    : ::testing::TestWithParam<IRDFTConstantAxesAndConstantSignalSizeTestParams> {};

TEST_P(IRDFTConstantAxesAndConstantSignalSizeTest, irdft_constant_axes_and_signal_size) {
    auto params = GetParam();

    auto input_shape = params.input_shape;
    set_shape_labels(input_shape, 10);
    auto data = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

    std::shared_ptr<op::v9::IRDFT> irdft;
    if (params.signal_size.empty()) {
        irdft = std::make_shared<op::v9::IRDFT>(data, axes_input);
    } else {
        auto signal_size_input =
            op::v0::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
        irdft = std::make_shared<op::v9::IRDFT>(data, axes_input, signal_size_input);
    }

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    EXPECT_EQ(irdft->get_output_partial_shape(0), params.ref_output_shape);
    EXPECT_EQ(get_shape_labels(irdft->get_output_partial_shape(0)), (params.expected_labels));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    IRDFTConstantAxesAndConstantSignalSizeTest,
    ::testing::Values(
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, 180},
                                                         {2, 0},
                                                         {},
                                                         {no_label, 11, 12}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{4, 180, 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {6, 180, 180},
                                                         {2, 0},
                                                         {},
                                                         {no_label, 11, 12}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{16, 500, 180, 369, 2},
                                                         {3},
                                                         Shape{},
                                                         {16, 998, 180, 369},
                                                         {0, 3, 1},
                                                         {},
                                                         {10, no_label, 12, 13}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), Dimension(12, 998)},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), Dimension(12, 998)},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), Dimension(7, 500), Dimension(12, 998)},
                                                         {1, 2},
                                                         {},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            Shape{},
            {Dimension(0, 2), Dimension(7, 500), Dimension(12, 998)},
            {1, 2},
            {},
            {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                         {2},
                                                         {2},
                                                         {2, 180, 77},
                                                         {1, 2},
                                                         {-1, 77},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                         {2},
                                                         {2},
                                                         {87, 180, 390},
                                                         {2, 0},
                                                         {390, 87},
                                                         {no_label, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{7, 50, 130, 400, 2},
                                                         {3},
                                                         {3},
                                                         {7, 40, 130, 600},
                                                         {3, 0, 1},
                                                         {600, -1, 40},
                                                         {10, no_label, 12, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                         {2},
                                                         {2},
                                                         {2, Dimension(0, 200), 77},
                                                         {1, 2},
                                                         {-1, 77},
                                                         {10, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                         {2},
                                                         {2},
                                                         {87, 180, 390},
                                                         {2, 0},
                                                         {390, 87},
                                                         {no_label, 11, no_label}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                                                         {3},
                                                         {3},
                                                         {Dimension(8, 129), 40, 130, 600},
                                                         {3, 0, 1},
                                                         {600, -1, 40},
                                                         {10, no_label, 12, no_label}}),
    PrintToDummyParamName());

TEST(type_prop, irdft_dynamic_axes) {
    auto input_shape = PartialShape{2, 180, 180, Dimension(1, 18)};
    set_shape_labels(input_shape, 10);
    const auto axes_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};

    auto data = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, axes_shape);
    auto irdft = std::make_shared<op::v9::IRDFT>(data, axes_input);

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    EXPECT_EQ(irdft->get_output_partial_shape(0), ref_output_shape);
    EXPECT_THAT(get_shape_labels(irdft->get_output_partial_shape(0)), Each(no_label));
}

struct IRDFTNonConstantAxesTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    PartialShape ref_output_shape;
};

struct IRDFTNonConstantAxesTest : ::testing::TestWithParam<IRDFTNonConstantAxesTestParams> {};

TEST_P(IRDFTNonConstantAxesTest, irdft_non_constant_axes) {
    auto params = GetParam();
    auto input_shape = params.input_shape;
    set_shape_labels(input_shape, 10);

    auto data = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);
    auto irdft = std::make_shared<op::v9::IRDFT>(data, axes_input);

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    EXPECT_EQ(irdft->get_output_partial_shape(0), params.ref_output_shape);
    EXPECT_THAT(get_shape_labels(irdft->get_output_partial_shape(0)), Each(no_label));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    IRDFTNonConstantAxesTest,
    ::testing::Values(
        IRDFTNonConstantAxesTestParams{{2, 180, 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, 180, Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), 180, 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, 180, 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}}),
    PrintToDummyParamName());

struct IRDFTNonConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<label_t> expected_labels;
};

struct IRDFTNonConstantSignalSizeTest : ::testing::TestWithParam<IRDFTNonConstantSignalSizeTestParams> {};

TEST_P(IRDFTNonConstantSignalSizeTest, irdft_non_constant_signal_size) {
    auto params = GetParam();

    auto input_shape = params.input_shape;
    set_shape_labels(input_shape, 10);

    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
    auto signal_size_input = std::make_shared<op::v0::Parameter>(element::i64, params.signal_size_shape);
    auto irdft = std::make_shared<op::v9::IRDFT>(data, axes_input, signal_size_input);

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    EXPECT_EQ(irdft->get_output_partial_shape(0), params.ref_output_shape);
    EXPECT_EQ(get_shape_labels(irdft->get_output_partial_shape(0)), params.expected_labels);
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    IRDFTNonConstantSignalSizeTest,
    ::testing::Values(IRDFTNonConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                           {2},
                                                           {2},
                                                           {2, Dimension(0, 200), Dimension::dynamic()},
                                                           {1, 2},
                                                           {no_label, no_label, no_label}},
                      IRDFTNonConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                           {2},
                                                           {2},
                                                           {Dimension::dynamic(), 180, Dimension(0, 400)},
                                                           {2, 0},
                                                           {no_label, no_label, no_label}},
                      IRDFTNonConstantSignalSizeTestParams{
                          {Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                          {3},
                          {3},
                          {Dimension(8, 129), Dimension::dynamic(), 130, Dimension(0, 500)},
                          {3, 0, 1},
                          {no_label, no_label, no_label, no_label}}),
    PrintToDummyParamName());

TEST(type_prop, irdft_invalid_input) {
    auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});

    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("The input rank must be greater or equal to 2"));

    data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("The last dimension of input data must be 2"));

    data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 2});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("The input rank must be greater than number of axes."));
}

TEST(type_prop, irdft_invalid_axes) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 2});

    auto axes = op::v0::Constant::create(element::i64, Shape{1}, {3});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("Axis value: 3, must be in range (-3, 2)"));

    axes = op::v0::Constant::create(element::i64, Shape{1}, {-3});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("Axis value: -3, must be in range (-3, 2)"));

    axes = op::v0::Constant::create(element::i64, Shape{2}, {0, -2});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("Each axis must be unique"));

    axes = op::v0::Constant::create(element::i64, Shape{1}, {2});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("Axis value: 2, must be in range (-3, 2)"));

    axes = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes),
                    Exception,
                    HasSubstr("Axes input must be 1D tensor."));
}

TEST(type_prop, irdft_invalid_signal_size) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 2});
    auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});

    auto signal_size = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes, signal_size),
                    Exception,
                    HasSubstr("Signal size input must be 1D tensor."));

    signal_size = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::IRDFT>(data, axes, signal_size),
                    Exception,
                    HasSubstr("Sizes of inputs 'axes' and 'signal_size' must be equal."));
}

TEST(type_prop, irdft_dynamic_types) {
    const auto input_shape = PartialShape{2, 180, 180, 2};
    const auto axes_shape = PartialShape::dynamic();
    const auto signal_size_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};

    auto data = std::make_shared<op::v0::Parameter>(element::dynamic, input_shape);
    auto axes_input = std::make_shared<op::v0::Parameter>(element::dynamic, axes_shape);
    auto signal_size_input = std::make_shared<op::v0::Parameter>(element::dynamic, signal_size_shape);
    auto irdft = std::make_shared<op::v9::IRDFT>(data, axes_input, signal_size_input);

    EXPECT_EQ(irdft->get_element_type(), element::dynamic);
    EXPECT_EQ(irdft->get_output_partial_shape(0), ref_output_shape);
}

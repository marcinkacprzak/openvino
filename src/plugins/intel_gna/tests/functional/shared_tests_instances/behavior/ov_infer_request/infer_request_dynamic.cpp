// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"

#include <vector>

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    OVInferRequestDynamicTests,
    ::testing::Combine(::testing::Values(ngraph::builder::subgraph::makeSplitConvConcat()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 4, 20, 20}, {1, 10, 18, 18}},
                           {{2, 4, 20, 20}, {2, 10, 18, 18}}}),
                       ::testing::Values(ov::test::utils::DEVICE_GNA),
                       ::testing::ValuesIn(configs)),
    OVInferRequestDynamicTests::getTestCaseName);

}  // namespace

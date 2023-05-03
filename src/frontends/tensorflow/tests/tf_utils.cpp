// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tf_utils.hpp"

#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::frontend;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace tests {

shared_ptr<Model> convert_model(const string& model_path,
                                const ConversionExtension::Ptr& conv_ext,
                                const vector<string>& input_names,
                                const vector<element::Type>& input_types,
                                const vector<PartialShape>& input_shapes) {
    FrontEndManager fem;
    auto front_end = fem.load_by_framework(TF_FE);
    if (!front_end) {
        throw "TensorFlow Frontend is not initialized";
    }
    if (conv_ext) {
        front_end->add_extension(conv_ext);
    }
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) + model_path);
    auto input_model = front_end->load(model_filename);
    if (!input_model) {
        throw "Input model is not read";
    }

    // set custom inputs, input shapes and types
    vector<Place::Ptr> input_places;
    for (const auto& input_name : input_names) {
        auto input_place = input_model->get_place_by_tensor_name(input_name);
        if (!input_place) {
            throw "Input place with name " + input_name + " is not found ";
        }
        input_places.push_back(input_place);
    }
    if (input_places.size() < input_types.size()) {
        throw "The number of input places is less than the number of types";
    }
    for (size_t ind = 0; ind < input_types.size(); ++ind) {
        auto input_type = input_types[ind];
        auto input_place = input_places[ind];
        input_model->set_element_type(input_place, input_type);
    }
    if (input_places.size() < input_shapes.size()) {
        throw "The number of input places is less than the number of shapes";
    }
    for (size_t ind = 0; ind < input_shapes.size(); ++ind) {
        auto input_shape = input_shapes[ind];
        auto input_place = input_places[ind];
        input_model->set_partial_shape(input_place, input_shape);
    }
    if (!input_places.empty()) {
        input_model->override_all_inputs(input_places);
    }

    auto model = front_end->convert(input_model);
    if (!model) {
        throw "Model is not converted";
    }

    return model;
}

}  // namespace tests
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
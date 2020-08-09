#ifndef SIMPLE_LAYER_H
#define SIMPLE_LAYER_H

#include "tensor.h"
// #include"../data-manipulation/instance.h"

using namespace std;
// using namespace data_manipulation;

namespace neural_network
{

    class simple_layer
    {
    private:
        int n_neurons;
        tensor weights;
        tensor bias;
        tensor deltas;

        // activation act_function;
    public:
        tensor output;
        // initialize a layer with n_neurons and input_dim as dims_n
        simple_layer(int n_neurons, int input_dim);
        // feed forward for first layer
        tensor feed_forward(vector<double> sample);
        tensor feed_forward(tensor input);
        tensor calculate_delta(tensor input, tensor loss);

        tensor get_weights();
        tensor get_bias();
        tensor get_output();
        // ~simple_layer();
    };

} // namespace neural_network

#endif
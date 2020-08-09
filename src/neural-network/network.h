#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "simple_layer.h"
#include "tensor.h"
#include "../data-manipulation/instance.h"

using namespace std;
using namespace data_manipulation;
using namespace neural_network;

namespace neural_network
{
    class network
    {
    private:
        vector<tensor> layers_output;

    public:
        vector<simple_layer> layers;

        network(vector<simple_layer> layers);
        tensor predict_one_sample(vector<double> sample);
        void backpropagation(vector<double> sample, vector<double> target);
        void backpropagation(vector<vector<double>> samples, vector<vector<double>> targets);
        // ~network();
    };

} // namespace neural_network

#endif
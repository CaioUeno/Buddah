#include "network.h"
#include "tensor.h"
#include <vector>
#include "simple_layer.h"
#include "../data-manipulation/instance.h"
#include <iostream>

using namespace std;
using namespace data_manipulation;
using namespace neural_network;

network::network(vector<simple_layer> layers_list)
{
    for (simple_layer layer : layers_list)
    {
        layers.push_back(layer);
    }
}

tensor network::predict_one_sample(vector<double> sample)
{

    tensor result;

    for (int i = 0; i < layers.size(); i++)
    {
        if (i == 0)
            result = layers[i].feed_forward(sample);

        else
            result = layers[i].feed_forward(result);

        // layers_output.push_back(result);
    }

    return result;
}

void network::backpropagation(vector<double> sample, vector<double> target)
{
    if (layers.size() == 1)
    {
        layers[0].feed_forward(tensor(sample, 1));
        cout << "out" << endl;
        layers[0].output.print_tensor();
        cout << "target" << endl;
        tensor ttt = tensor(target, 1);
        ttt.print_tensor();
        layers[0].calculate_delta(tensor(sample, 1), layers[0].output - ttt);
    }
}

void network::backpropagation(vector<vector<double>> samples, vector<vector<double>> targets)
{
    if (samples.size() != targets.size())
    {
        exit(EXIT_FAILURE);
    }
    
    if (layers.size() == 1)
    {
        for (int i = 0; i < samples.size(); i++)
        {
            layers[0].feed_forward(tensor(samples[i], 1));
            cout << "out" << endl;
            layers[0].output.print_tensor();
            cout << "target" << endl;
            tensor ttt = tensor(targets[i], 1);
            ttt.print_tensor();
            layers[0].calculate_delta(tensor(samples[i], 1), layers[0].output - ttt);
        }
    }
}
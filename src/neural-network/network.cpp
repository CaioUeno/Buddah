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

    }

    return result;
}

void network::backpropagation(vector<double> sample, vector<double> target)
{
    if (layers.size() <= 3)
    {

        predict_one_sample(sample);
        // cout << "here" << endl;
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            // cout << i << endl;

            if (1 == layers.size())
            {
                tensor ttt = tensor(target, 1);
                // cout << "target" << endl;
                // ttt.print_tensor();
                layers[i].calculate_delta(tensor(sample, 1), layers[i].output - ttt);
                break;
            }
            else
            {
                if (i == 0)
                    layers[i].calculate_delta(tensor(sample, 1), layers[i + 1]);

                else if (i == layers.size() - 1)
                {
                    tensor ttt = tensor(target, 1);
                    // cout << "target" << endl;
                    // ttt.print_tensor();
                    layers[i].calculate_delta(layers[i - 1].output, layers[i].output - ttt);
                }
                else
                {
                    layers[i].calculate_delta(layers[i - 1].output, layers[i + 1]);
                }
            }
        }
    }

    tensor ttt = tensor(target, 1);
    cout << "target" << endl;
    ttt.print_tensor();
    cout << "out" << endl;
    layers[layers.size()-1].output.print_tensor();
}

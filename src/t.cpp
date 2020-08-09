#include <iostream>
#include "data-manipulation/dataset.h"
#include "neural-network/tensor.h"
#include "neural-network/simple_layer.h"
#include "neural-network/network.h"

using namespace std;
using namespace neural_network;

int main(int argc, char const *argv[])
{
    // time(NULL);
    simple_layer layer_1(1, {4});
    network net({layer_1});
    dataset ds("data-manipulation/iris.txt");

    for (int epochs = 0; epochs < 100; epochs++)
    {
        for (int i = 0; i < ds.get_shape()[0]; i++)
        {
            net.backpropagation(ds.get_instance(i).get_features(), ds.get_instance(i).get_target());
        }
    }

    return 0;
}

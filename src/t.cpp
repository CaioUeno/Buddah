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
    simple_layer layer_1(2, 4);
    simple_layer layer_2(4, 4);
    simple_layer layer_3(4, 1);
    network net({layer_3});
    dataset ds("data-manipulation/iris.txt");

    

    for (int epochs = 0; epochs < 30; epochs++)
    {
        for (int i = 0; i < ds.get_shape()[0]; i++)
        {
            net.backpropagation(ds.get_instance(i).get_features(), ds.get_instance(i).get_target());
        }
    }
    return 0;

}
#include "simple_layer.h"
#include <iostream>
#include <math.h>
#include "tensor.h"

using namespace std;
using namespace neural_network;
// using namespace data_manipulation;

double sigmoid(double d)
{
    return 1 / (1 + exp(-d));
}

tensor sigmoid_d(vector<double> array)
{
    vector<double> result;
    for (double d : array)

        result.push_back(sigmoid(d) * (1 - sigmoid(d)));

    return tensor(result, 1);
}

simple_layer::simple_layer(int input_dim, int n_neurons)
{

    if (input_dim < 1)
    {
        cout << "simple layer cannot have non-positive dimensions." << endl;
        exit(EXIT_FAILURE);
    }

    else
    {
        weights = tensor({n_neurons, input_dim}, 0, 1);
        bias = tensor({n_neurons, 1}, 0, 1);
        delta = tensor({n_neurons, input_dim}, 0.0);

        this->n_neurons = n_neurons;
    }
}

tensor simple_layer::feed_forward(vector<double> sample)
{
    tensor tensor_input = tensor(sample, 1).transpose();
    tensor result = (weights * tensor_input) + bias;
    vector<int> shape = result.get_shape();

    for (int i = 0; i < shape[0]; i++)
        result.values[i][0] = sigmoid(result.values[i][0]);

    output = result;

    return result;
}

tensor simple_layer::feed_forward(tensor input)
{
    tensor result = (weights * input) + bias;
    vector<int> shape = result.get_shape();

    for (int i = 0; i < shape[0]; i++)
        result.values[i][0] = sigmoid(result.values[i][0]);

    output = result;

    return result;
}


tensor simple_layer::calculate_delta(tensor input, tensor loss)
{
    delta = sigmoid_d(output.values[0]).mul_element_wise(loss);
    // input.print_tensor();
    // delta.print_tensor();
    vector<vector<double>> a;
    for (int i = 0; i < delta.get_shape()[1]; i++)
        a.push_back(input.scalar_mul(delta.get_column(i)[0]).get_column(0));

    tensor b = tensor(a).scalar_mul(0.1);
    // b.print_tensor();
    // weights.print_tensor();
    weights = (weights - b);
    return delta;
}

tensor simple_layer::calculate_delta(tensor input, simple_layer next_layer)
{

    tensor next_layer_delta = next_layer.get_delta();
    tensor next_layer_weights = next_layer.get_weights();
    // next_layer_delta.print_tensor();
    // next_layer_weights.print_tensor();

    delta = (next_layer_delta * next_layer_weights);
    
    
    double sum = 0;
    for (double d : delta.get_row(0))
    {
        sum += d;
    }
    
    delta = sigmoid_d(output.get_column(0));
    // delta.shape[1] = delta.get_row(0).size();
    vector<vector<double>> a;
    // delta.print_tensor();
    for (int i = 0; i < delta.get_shape()[1]; i++)
    {
        // input.scalar_mul(delta.get_column(i)[0]).print_tensor();
        a.push_back(input.scalar_mul(delta.get_column(i)[0]).get_row(0));
    }


    tensor b = tensor(a).scalar_mul(0.1);
    // b.print_tensor();
    // weights.print_tensor();
    weights = (weights - b);
    return delta;
}

tensor simple_layer::get_weights()
{
    return weights;
}

tensor simple_layer::get_bias()
{
    return bias;
}

tensor simple_layer::get_delta()
{
    return delta;
}

tensor simple_layer::get_output()
{
    return output;
}
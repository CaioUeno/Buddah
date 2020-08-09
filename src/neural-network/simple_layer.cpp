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

simple_layer::simple_layer(int n_neurons, int input_dim)
{

    if (input_dim < 1)
    {
        cout << "simple layer cannot have non-positive dimensions." << endl;
        exit(EXIT_FAILURE);
    }

    else
    {
        weights = tensor({input_dim, n_neurons}, 0, 1);
        bias = tensor({1, n_neurons}, 0, 1);
        deltas = tensor({input_dim, n_neurons}, 0.0);

        this->n_neurons = n_neurons;
    }
}

tensor simple_layer::feed_forward(vector<double> sample)
{

    tensor result = (tensor(sample, 1) * weights) + bias;
    vector<int> shape = result.get_shape();

    for (int i = 0; i < shape[1]; i++)
        result.values[0][i] = sigmoid(result.values[0][i]);

    output = result;

    return result;
}

tensor simple_layer::feed_forward(tensor input)
{

    tensor result = (input * weights) + bias;
    vector<int> shape = result.get_shape();

    for (int i = 0; i < shape[1]; i++)
        result.values[0][i] = sigmoid(result.values[0][i]);

    output = result;

    return result;
}

tensor simple_layer::calculate_delta(tensor input, tensor loss)
{
    deltas = sigmoid_d(output.values[0]) * loss;
    double a = deltas.values[0][0];
    tensor b = input.scalar_mul(a).scalar_mul(0.1);

    // b.row_to_column();
    // b.print_tensor();
    tensor ssss = b.transpose();
    // ssss.print_tensor();
    weights = (weights - ssss);
    return deltas;
}

tensor simple_layer::get_weights()
{
    return weights;
}

tensor simple_layer::get_bias()
{
    return bias;
}

tensor simple_layer::get_output()
{
    return output;
}
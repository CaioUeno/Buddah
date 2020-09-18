#include "tensor.h"
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace neural_network;

tensor::tensor()
{
}

tensor::tensor(vector<int> shape, double mean, double var)
{
    srand((unsigned)time(NULL));
    default_random_engine generator(rand());
    normal_distribution<double> distribution(0, 1); // pseudo normal distribution

    for (int i = 0; i < shape[0]; i++)
    {

        vector<double> dist;
        for (int j = 0; j < shape[1]; j++)
        {

            double generated_number = distribution(generator);
            while (generated_number > 1 || generated_number < -1) // ensure that generated number is in [-1, 1]
                generated_number = distribution(generator);

            dist.push_back(generated_number);
        }

        values.push_back(dist);
    }

    this->shape = shape;
}

tensor::tensor(vector<int> shape, double value)
{

    for (int i = 0; i < shape[0]; i++)
    {

        vector<double> dist;
        for (int j = 0; j < shape[1]; j++)
            dist.push_back(value);

        values.push_back(dist);
    }

    this->shape = shape;
}

tensor::tensor(vector<double> vals, int n)
{

    for (int i = 0; i < n; i++)
        values.push_back(vals);

    int columns = vals.size();
    this->shape = {n, columns};
}

tensor::tensor(vector<vector<double>> rows)
{
    for (vector<double> row : rows)
        values.push_back(row);
    int n_rows = rows.size(), n_columns = rows[0].size();
    this->shape = {n_rows, n_columns};
}

tensor tensor::operator+(tensor &other_tensor)
{

    if (shape[0] != other_tensor.get_shape()[0] || shape[1] != other_tensor.get_shape()[1])
    {
        cout << "Tensors have different shapes." << endl;
        exit(EXIT_FAILURE);
    }

    else
    {

        tensor result(get_shape(), 0);

        for (int i = 0; i < shape[0]; i++)
            transform(values[i].begin(), values[i].end(), other_tensor.values[i].begin(), result.values[i].begin(), plus<double>());

        return result;
    }
}

tensor tensor::operator-(tensor &other_tensor)
{

    if (shape[0] != other_tensor.get_shape()[0] || shape[1] != other_tensor.get_shape()[1])
    {
        cout << "Tensors have different shapes." << endl;
        exit(EXIT_FAILURE);
    }

    else
    {

        tensor result(get_shape(), 0);

        for (int i = 0; i < shape[0]; i++)
            transform(values[i].begin(), values[i].end(), other_tensor.values[i].begin(), result.values[i].begin(), minus<double>());

        return result;
    }
}

tensor tensor::operator*(tensor &other_tensor)
{

    if (shape[1] != other_tensor.get_shape()[0])
    {
        cout << "Cannot multiply tensors with shapes (" << shape[0] << ", " << shape[1] << ") and (" << other_tensor.get_shape()[0] << ", " << other_tensor.get_shape()[1] << ")" << endl;
        exit(EXIT_FAILURE);
    }

    else
    {
        tensor result({shape[0], other_tensor.get_shape()[1]}, 0.0);

        int n_lines = shape[0];
        int n_columns = other_tensor.get_shape()[1];

        for (int i = 0; i < n_lines; i++)
        {

            for (int j = 0; j < n_columns; j++)
            {

                vector<double> line = values[i];
                vector<double> column = other_tensor.get_column(j);
                vector<double> aux(line.size());

                transform(line.begin(), line.end(), column.begin(), aux.begin(), multiplies<double>());
                result.values[i][j] = accumulate(aux.begin(), aux.end(), decltype(aux)::value_type(0));
            }
        }
        return result;
    }
}

tensor tensor::mul_element_wise(tensor &other_tensor)
{

    if (shape[0] != other_tensor.get_shape()[0] || shape[1] != other_tensor.get_shape()[1])
    {
        cout << "Tensors have different shapes." << endl;
        exit(EXIT_FAILURE);
    }

    else
    {

        tensor result(get_shape(), 0);

        for (int i = 0; i < shape[0]; i++)
            transform(values[i].begin(), values[i].end(), other_tensor.values[i].begin(), result.values[i].begin(), multiplies<double>());

        return result;
    }
}

tensor tensor::scalar_mul(double scalar)
{

    tensor result = *this;

    for (int i = 0; i < this->shape[0]; i++)
        for (int j = 0; j < this->shape[1]; j++)
            result.values[i][j] *= scalar;

    return result;
}

vector<double> tensor::get_column(int index)
{

    if (index > values[0].size())
    {
        cout << "Tensor does not have column " << index << ". Tensor has " << values[0].size() << " column(s)." << endl;
        exit(EXIT_FAILURE);
    }

    else
    {
        vector<double> column;

        for (int i = 0; i < shape[0]; i++)
            column.push_back(values[i][index]);

        return column;
    }
}

vector<double> tensor::get_row(int index)
{

    if (index > values.size())
    {
        cout << "Tensor does not have row " << index << ". Tensor has " << values.size() << " row(s)." << endl;
        exit(EXIT_FAILURE);
    }

    else
        return values[index];
}

tensor tensor::transpose()
{
    vector<vector<double>> rows;
    for (int i = 0; i < shape[1]; i++)
        rows.push_back(get_column(i));

    return tensor(rows);
}

vector<int> tensor::get_shape()
{
    return shape;
}

void tensor::print_tensor()
{
    for (vector<double> l : values)
    {
        for (double i : l)
            cout << i << " ";
        cout << endl;
    }
}
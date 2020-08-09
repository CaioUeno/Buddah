#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

using namespace std;
namespace neural_network
{

    class tensor
    {
    private:
        

    public:
        //  essentially a matrix
        vector<vector<double>> values;
        vector<int> shape;

        tensor();
        // initialize a tensor with shape and normal distribution~(mean, var)
        tensor(vector<int> shape, double mean, double var);
        // initialize a tensor with shape and all values equal to value
        tensor(vector<int> shape, double value);
        // create a tensor with n equal rows (vals) 
        tensor(vector<double> vals, int n);
        //  create a tensor with rows
        tensor(vector<vector<double>> rows);
        // element wise sum
        tensor operator+(tensor &other_tensor);
        // element wise subtraction
        tensor operator-(tensor &other_tensor);
        // matrix multiplicaiton
        tensor operator*(tensor &other_tensor);
        // element wise multiplication
        tensor mul_element_wise(tensor &other_tensor);
        // scalar multiplication
        tensor scalar_mul(double scalar);
        // returns the column at index
        vector<double> get_column(int index);
        // return the row at index
        vector<double> get_row(int index);
        // return a NEW tensor with transpose values
        tensor transpose();

        vector<int> get_shape();
        void print_tensor();
        // ~tensor();
    };

} // namespace neural_network

#endif
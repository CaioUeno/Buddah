#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <vector>

#include "instance.h"

using namespace std;
namespace data_manipulation
{

    class dataset
    {
    private:
        vector<instance> instances;
        vector<int> shape;

    public:
        dataset(string filename);
        vector<int> get_shape();
        instance get_instance(int index);
        vector<instance> get_instances();
        // ~dataset();
    };

} // namespace data_manipulation

#endif
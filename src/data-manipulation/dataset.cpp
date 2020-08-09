#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>
#include "dataset.h"
// #include "instance.h"

using namespace std;
using namespace data_manipulation;

dataset::dataset(string filename)
{

    ifstream file;
    file.open(filename); // open the dataset file

    if (!file.is_open())
        cout << "File error: " << filename << " could not be open.";

    else
    {
        int number_of_lines = 0;
        string line;

        while (getline(file, line))
        {
            number_of_lines++;

            istringstream iss(line);

            vector<double> feat_target{istream_iterator<double>(iss), istream_iterator<double>()};
            vector<double> features(feat_target.begin(), feat_target.end() - 1);
            vector<double> target(feat_target.end() - 1, feat_target.end());

            instance i(features, target);
            instances.push_back(i);
        }

        shape.push_back(number_of_lines);

        if (number_of_lines == 0)
        {
            cout << "Dataset has no instance - check file " << filename << "." << endl;
            exit(EXIT_FAILURE);
        }
    }
}

vector<int> dataset::get_shape()
{
    return shape;
}

instance dataset::get_instance(int index)
{
    return instances[index];
}

vector<instance> dataset::get_instances(){
    return instances;
}
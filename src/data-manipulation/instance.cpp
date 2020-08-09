#include <iostream>
#include "instance.h"

using namespace std;
using namespace data_manipulation;

instance::instance(vector<double> features, vector<double> target)
{
    this->attributes = features;
    this->target = target;
}

vector<double> instance::get_features()
{
    return vector<double>(attributes);
}

vector<double> instance::get_target()
{
    return vector<double>(target);
}

void instance::print_features()
{
    for (double i : attributes)
        cout << i << endl;
}
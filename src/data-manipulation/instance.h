#ifndef INSTANCE_H
#define INSTANCE_H

#include <vector>

using namespace std;

namespace data_manipulation
{

    class instance
    {
    private:
        vector<double> attributes;
        vector<double> target;

    public:
        instance(vector<double> features, vector<double> target);
        vector<double> get_features();
        vector<double> get_target();
        
        void print_features();
        //~instance();
    };

} // namespace data_manipulation

#endif
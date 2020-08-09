#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>

using namespace std;

namespace neural_network
{
    class activation
    {
    private:
        //  there is no attributes
    public:
        activation();
        vector<double> apply();
        // ~activation();
    };

} // namespace neural_network

#endif
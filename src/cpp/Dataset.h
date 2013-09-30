#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;

class Dataset  
{
    private:
        float **examples;
        float *classifications;

    public:
        Dataset();
        ~Dataset();
        void load(char *);
        float *  get_example(int index);
        float get_classifications(int index);
};

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include "Dataset.h"

using namespace std;

class HebbianAutoencoder   
{
    private:
        int hidden;
        int input;

        bool tied;

        float **weights_encoder;
        float **weights_decoder;

        float *bias_encoder;
        float *bias_decoder;

        float *hidden_activations;
        float *output_activations;

        Dataset training;

        float learning_rate;

    public:
        HebbianAutoencoder();
        ~HebbianAutoencoder();
        float monitor(float *inputs, float *outputs);
        float train();
        void save();
        float sigmoid(float);
        float compute(float * inputs, float * weights, float * bias, int length);
};

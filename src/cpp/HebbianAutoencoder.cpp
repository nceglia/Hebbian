#include "HebbianAutoencoder.h"

HebbianAutoencoder::HebbianAutoencoder()  {
    hidden = 100;
    input = 784;

    tied = false;

    srand(time(NULL));

    weights_encoder = new float*[hidden];
    hidden_activations = new float[hidden];
    for(int i = 0; i < hidden; i++)  {
        weights_encoder[i] = new float[input];
        hidden_activations[i] = 0.0;
    }
    weights_decoder = new float*[input];
    output_activations = new float[input];
    for(int i = 0; i < input;i++)  {
        weights_decoder[i] = new float[hidden];
        output_activations[i] = 0.0;
    }

    bias_encoder = new float[input];
    bias_decoder = new float[hidden];

    training.load("mnist_training.csv");

    for(int i = 0; i < hidden; i++)  {
        bias_encoder[i] = (float)(rand()%200-100)/100.0;
        bias_decoder[i] = (float)(rand()%200-100)/100.0;
        for(int j = 0; j < input; j++)
            weights_encoder[i][j] = (float)(rand()%200-100)/100.0;
    }
    
    for(int i = 0; i < input; i++)  {
        for(int j = 0; j < hidden; j++)
            weights_decoder[i][j] = (float)(rand()%200-100)/100.0;
    }
    learning_rate = 0.0001;
}

HebbianAutoencoder::~HebbianAutoencoder() {

}

float HebbianAutoencoder::monitor(float * inputs, float * outputs)  {
    float error = 0.0;
    for(int i = 0; i < input; i++) {
        error += outputs[i] - inputs[i]; 
    }
    return error;
}

float HebbianAutoencoder::train()  {
    for(int example = 0; example < 60000; example++) {
        float * features = training.get_example(example);
        for(int i = 0; i < hidden; i++)  {
            hidden_activations[i] = sigmoid(compute(features,weights_encoder[i],bias_encoder,input));
            for(int j = 0; j < input; j++) {
                float delta = learning_rate * hidden_activations[i] * features[j];
                if ( delta != 0.0 ) {
                    weights_encoder[i][j] += delta;
                    weights_encoder[i][j] =  sigmoid(weights_encoder[i][j]);
                }
            }
        }
        for(int i = 0; i < input; i++) {
            output_activations[i] = compute(hidden_activations,weights_decoder[i],bias_decoder,hidden);
            for(int j = 0; j < hidden; j++)  {
                float delta = learning_rate * output_activations[i] * hidden_activations[j];
                if(delta != 0.0)  {
                    weights_decoder[i][j] += delta;
                    weights_decoder[i][j] =  sigmoid(weights_decoder[i][j]);
                }
            }
        }
        float current = monitor(features,output_activations);
        cout << "\tExample:  " << current << endl;
        if (example%50 == 0.0)
            cout << "\t\tSaving Model" << endl;
            save();
    }
}

float HebbianAutoencoder::sigmoid(float value)  {
    return 1.0/(1.0+exp(1.0*value));
}

float HebbianAutoencoder::compute(float * inputs, float * weights, float * bias, int length)  {
    float activation = 0.0;
    for(int i = 0; i < length; i++) {
        activation += inputs[i]*weights[i] + bias[i];
    }
    return activation;
}

void HebbianAutoencoder::save()  {
    ofstream model;
    model.open("model_c.txt");
    for(int i = 0; i < hidden; i++)  {
        for(int j = 0; j < input; j++)
            model << weights_encoder[i][j] << "\t";
        model << "\n";
    }
    
    for(int i = 0; i < input; i++)  {
        for(int j = 0; j < hidden; j++)
            model<< weights_decoder[i][j] << "\t";
        model << "\n";
    }
    model.close();
}







































package com.company.train.trainer;

import com.company.train.TestSet;
import com.company.model.Matrix;
import com.company.model.network.NeuralNetwork;

public abstract class Trainer {
    NeuralNetwork network;
    final TestSet testSet;

    double[] lastTrainErrorsData;

    final TrainerOptions options;

    // Inertia calculating basing on this increments
    Matrix[] previousWeightsIncrements;
    Matrix[] previousBiasesIncrements;

    public Trainer(NeuralNetwork network, TestSet testSet, TrainerOptions options) {
        this.network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
        this.testSet = testSet;
        this.options = options;
    }

    public static double calcOutputError(double[] output, double[] expectedOutput) {
        double outputError = 0;

        for (int i = 0; i < output.length; ++i) {
            outputError += (output[i] - expectedOutput[i]) * (output[i] - expectedOutput[i]);
        }

        return outputError;
    }

    public double[] getLastTrainErrorsData() {
        return lastTrainErrorsData;
    }

    // Trains network by tweaking its parameters after all tests' errors calculated
    public abstract NeuralNetwork trainNetwork();

    NeuralNetwork tweakNetworkParametersByGradients(NeuralNetwork network, double[] weightsErrorGradient, double[] biasesErrorGradient) {
        // Weights tweak
        int currentWeightIndex = 0;
        
        for (int i = 0; i < network.weights.length; ++i) {
            for (int j = 0; j < network.weights[i].N; ++j) {
                for (int k = 0; k < network.weights[i].M; ++k, ++currentWeightIndex) {
                    network.weights[i].values[j][k] = network.weights[i].values[j][k] - options.learnSpeed() * weightsErrorGradient[currentWeightIndex];
                    network.weights[i].values[j][k] += options.inertiaCoefficient() * previousWeightsIncrements[i].values[j][k];
                }
            }
        }

        // Biases tweak
        int currentBiasIndex = 0;
        
        for (int i = 0; i < network.biases.length; ++i) {
            for (int j = 0; j < network.biases[i].N; ++j, ++currentBiasIndex) {
                network.biases[i].values[j][0] = network.biases[i].values[j][0] - options.learnSpeed() * biasesErrorGradient[currentBiasIndex];
                network.biases[i].values[j][0] += options.inertiaCoefficient() * previousBiasesIncrements[i].values[j][0];
            }
        }

        return network;
    }
}

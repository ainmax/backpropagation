package com.company.train.trainer;

import com.company.model.Matrix;
import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;
import com.company.train.gradient.BiasesOutputErrorGradient;
import com.company.train.gradient.WeightsOutputErrorGradient;

public class OnlineTrainer extends Trainer{
    public OnlineTrainer(NeuralNetwork network, TestSet testSet, TrainerOptions options) {
        super(network, testSet, options);
    }

    NeuralNetwork trainEpoch(NeuralNetwork network) {
        // Define zero increments
        previousWeightsIncrements = new Matrix[network.weights.length];
        previousBiasesIncrements = new Matrix[network.biases.length];

        for (int i = 0; i < network.weights.length; ++i) {
            previousWeightsIncrements[i] = new Matrix(network.weights[i].N, network.weights[i].M);
        }

        for (int i = 0; i < network.biases.length; ++i) {
            previousBiasesIncrements[i] = new Matrix(network.biases[i].N, network.biases[i].M);
        }

        testSet.clearTestsQueue();

        double[] currentWeightGradient;
        double[] currentBiasGradient;

        while (testSet.hasNextTest()) {
            TestSet.Test currentTest = testSet.nextTest();

            for (int i = 0; i < 2; ++i) {
                // Save old parameters
                Matrix[] oldWeights = new Matrix[network.weights.length];
                Matrix[] oldBiases = new Matrix[network.biases.length];

                for (int p = 0; p < network.weights.length; ++p) {
                    oldWeights[p] = new Matrix(network.weights[p]);
                }

                for (int p = 0; p < network.biases.length; ++p) {
                    oldBiases[p] = new Matrix(network.biases[p]);
                }

                // Calculate output error
                currentWeightGradient = (new WeightsOutputErrorGradient(network, currentTest).getOutputErrorGradient()).values[0];
                currentBiasGradient = (new BiasesOutputErrorGradient(network, currentTest).getOutputErrorGradient()).values[0];

                // Tweak network's parameters
                tweakNetworkParametersByGradients(network, currentWeightGradient, currentBiasGradient);

                // Calculate new increments of weights
                for (int p = 0; p < network.weights.length; ++p) {
                    for (int j = 0; j < network.weights[p].N; ++j) {
                        for (int k = 0; k < network.weights[p].M; ++k) {
                            previousWeightsIncrements[p].values[j][k] = network.weights[p].values[j][k] - oldWeights[p].values[j][k];
                        }
                    }
                }

                // Calculate new increments of biases
                for (int p = 0; p < network.biases.length; ++p) {
                    for (int j = 0; j < network.biases[p].N; ++j) {
                        previousBiasesIncrements[p].values[j][0] = network.biases[p].values[j][0] - oldBiases[p].values[j][0];
                    }
                }
            }
        }

        return network;
    }
}

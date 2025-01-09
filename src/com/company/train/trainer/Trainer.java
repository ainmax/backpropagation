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

    // Trains network by tweaking its parameters
    public NeuralNetwork trainNetwork() {
        boolean isNetworkTrainedEnough = false;

        do {
            lastTrainErrorsData = new double[options.trainEpochsCount()];

            for (int i = 0; i < options.trainEpochsCount(); ++i) {
                // Tweak network's parameters
                network = trainEpoch(new NeuralNetwork(network));

                // Average error calculating
                testSet.clearTestsQueue();

                double maxOutputError = 0;
                double averageOutputError = 0;

                do {
                    TestSet.Test currentTest = testSet.nextTest();
                    double error = calcOutputError(network.calcOutputBy(currentTest.input()), currentTest.correctOutput());
                    averageOutputError += error;
                    maxOutputError = Math.max(maxOutputError, error);
                } while (testSet.hasNextTest());

                averageOutputError /= testSet.size;
                lastTrainErrorsData[i] = averageOutputError;

                System.out.printf("%f -- %f%n", averageOutputError, maxOutputError);

                if (averageOutputError <= options.maxAcceptableAverageOutputError() && maxOutputError <= options.maxAcceptableOutputError()) {
                    isNetworkTrainedEnough = true;
                    break;
                }
            }

            System.out.println(lastTrainErrorsData[lastTrainErrorsData.length - 1]);

            // If network wasn't trained enough to have average error smaller than maximal acceptable error - training restarts
            if (!isNetworkTrainedEnough) {
                network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
            }
        } while (!isNetworkTrainedEnough);

        return network;
    }

    abstract NeuralNetwork trainEpoch(NeuralNetwork network);

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

package com.company.train;

import com.company.model.BiasesGradient;
import com.company.model.NeuralNetwork;
import com.company.model.WeightsGradient;

public class Trainer {
    private NeuralNetwork network;
    private final TestsBase testsBase;

    double[] lastTrainErrorsData;

    private static final double LEARN_SPEED = 0.1;
    private static final double COMPRESSION_COEFFICIENT = 1;

    public Trainer(NeuralNetwork network, TestsBase testsBase) {
        this.network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
        this.testsBase = testsBase;
    }

    private static double calcOutputError(double[] output, double[] expectedOutput) {
        double outputError = 0;

        for (int i = 0; i < output.length; ++i) {
            outputError += (output[i] - expectedOutput[i]) * (output[i] - expectedOutput[i]);
        }

        return outputError;
    }

    public NeuralNetwork trainNetworkOffline(int trainEpochsQuantity, double maxAcceptableAverageOutputError) {
        boolean isNetworkTrainedEnough = false;

        do {
            double averageOutputError = 0;

            for (int i = 0; i < trainEpochsQuantity; ++i) {
                 network = trainEpochOffline(new NeuralNetwork(network));
            }

            // !!!!!!!!!!!!!!!!!!!!!!! There should be network error research !!!!!!!!!!!!!!!!!!!!!!!!!!!

            if (averageOutputError > maxAcceptableAverageOutputError) {
                network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
            } else {
                isNetworkTrainedEnough = true;
            }
        } while (!isNetworkTrainedEnough);

        return network;
    }

    // Reweight network by gradients and returns average error
    private NeuralNetwork trainEpochOffline(NeuralNetwork network) {
        double[][] weightsErrorGradients = new double[testsBase.size][];
        double[][] biasesErrorGradients = new double[testsBase.size][];

        // Define gradient for each test
        while (testsBase.hasNextTest()) {
            TestsBase.Test currentTest = testsBase.nextTest();
            int i = testsBase.getCurrentTestIndex();

            weightsErrorGradients[i] = new WeightsGradient(network, currentTest).getErrorFunctionGradient();
            biasesErrorGradients[i] = new BiasesGradient(network, currentTest).getErrorFunctionGradient();
        }

        testsBase.clearTestsQueue();

        // Calculating sum of weights error gradients
        double[] averageWeightsErrorGradient = weightsErrorGradients[0];

        for (int i = 1; i < weightsErrorGradients.length; ++i) {
            for (int j = 0; j < weightsErrorGradients.length; ++j) {
                averageWeightsErrorGradient[j] += weightsErrorGradients[i][j];
            }
        }
        
        // Calculating sum of biases error gradients
        double[] averageBiasesErrorGradient = biasesErrorGradients[0];

        for (int i = 1; i < biasesErrorGradients.length; ++i) {
            for (int j = 0; j < biasesErrorGradients.length; ++j) {
                averageWeightsErrorGradient[j] += weightsErrorGradients[i][j];
                averageBiasesErrorGradient[j] += biasesErrorGradients[i][j];
            }
        }

        // Calculating average gradients by all tests
        for (int i = 0; i < averageWeightsErrorGradient.length; ++i) {
            averageWeightsErrorGradient[i] /= testsBase.size;
        }

        for (int i = 0; i < averageBiasesErrorGradient.length; ++i) {
            averageBiasesErrorGradient[i] /= testsBase.size;
        }

        return tweakNetworkParametersByGradients(network, averageWeightsErrorGradient, averageBiasesErrorGradient);
    }

    // Returns average error after every epoch
    public NeuralNetwork trainNetworkOnline(int trainEpochsQuantity, double maxAcceptableAverageOutputError) {
        boolean isNetworkTrainedEnough = false;

        do {
            double averageOutputError = 0;

            for (int i = 0; i < trainEpochsQuantity; ++i) {
                network = trainEpochOnline(new NeuralNetwork(network));
            }

            // !!!!!!!!!!!!!!!!!!!!!!! There should be network error research !!!!!!!!!!!!!!!!!!!!!!!!!!!

            if (averageOutputError > maxAcceptableAverageOutputError) {
                network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
            } else {
                isNetworkTrainedEnough = true;
            }
        } while (!isNetworkTrainedEnough);

        return network;
    }

    private NeuralNetwork trainEpochOnline(NeuralNetwork network) {
        // Calculating gradient for each test
        double[] currentWeightGradient;
        double[] currentBiasGradient;

        while (testsBase.hasNextTest()) {
            TestsBase.Test currentTest = testsBase.nextTest();
            currentWeightGradient = new WeightsGradient(network, currentTest).getErrorFunctionGradient();
            currentBiasGradient = new BiasesGradient(network, currentTest).getErrorFunctionGradient();

            tweakNetworkParametersByGradients(network, currentWeightGradient, currentBiasGradient);
        }

        testsBase.clearTestsQueue();

        return network;
    }

    private NeuralNetwork tweakNetworkParametersByGradients(NeuralNetwork network, double[] weightsErrorGradient, double[] biasesErrorGradient) {
        // Weights tweak
        int currentWeightIndex = 0;
        
        for (int i = 0; i < network.weights.length; ++i) {
            for (int j = 0; j < network.weights[i].N; ++j) {
                for (int k = 0; k < network.weights[i].M; ++k, ++currentWeightIndex) {
                    network.weights[i].values[j][k] = COMPRESSION_COEFFICIENT * network.weights[i].values[j][k] - LEARN_SPEED * weightsErrorGradient[currentWeightIndex];
                }
            }
        }

        // Biases tweak
        int currentBiasIndex = 0;
        
        for (int i = 0; i < network.biases.length; ++i) {
            for (int j = 0; j < network.biases[i].N; ++j, ++currentBiasIndex) {
                network.biases[i].values[j][0] = COMPRESSION_COEFFICIENT * network.biases[i].values[j][0] - LEARN_SPEED * biasesErrorGradient[currentBiasIndex];
            }
        }

        return network;
    }
}

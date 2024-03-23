package com.company.train;

import com.company.model.BiasesOutputErrorGradient;
import com.company.model.NeuralNetwork;
import com.company.model.WeightsOutputErrorGradient;

public class Trainer {
    private NeuralNetwork network;
    private final TestsBase testsBase;

    private double[] lastTrainErrorsData;

    private static final double LEARN_SPEED = 0.1;
    private static final double COMPRESSION_COEFFICIENT = 1;

    public Trainer(NeuralNetwork network, TestsBase testsBase) {
        this.network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
        this.testsBase = testsBase;
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
    public NeuralNetwork trainNetworkOffline(int trainEpochsQuantity, double maxAcceptableAverageOutputError) {
        boolean isNetworkTrainedEnough = false;

        do {
            double lastAverageOutputError = 0;
            lastTrainErrorsData = new double[trainEpochsQuantity];

            for (int i = 0; i < trainEpochsQuantity; ++i) {
                 network = trainEpochOffline(new NeuralNetwork(network));

                // Average error calculating
                testsBase.clearTestsQueue();

                do {
                    TestsBase.Test currentTest = testsBase.nextTest();
                    lastAverageOutputError += calcOutputError(network.calcOutputBy(currentTest.input()), currentTest.correctOutput());
                } while (testsBase.hasNextTest());

                lastAverageOutputError /= testsBase.size;
                lastTrainErrorsData[i] = lastAverageOutputError;
            }

            System.out.println(lastAverageOutputError);

            // If network wasn't trained enough to have average error smaller than maximal acceptable error - training restarts
            if (lastAverageOutputError > maxAcceptableAverageOutputError) {
                network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
            } else {
                isNetworkTrainedEnough = true;
            }
        } while (!isNetworkTrainedEnough);

        return network;
    }

    private NeuralNetwork trainEpochOffline(NeuralNetwork network) {
        double[][] weightsErrorGradients = new double[testsBase.size][];
        double[][] biasesErrorGradients = new double[testsBase.size][];

        // Define gradient for each test
        testsBase.clearTestsQueue();

        while (testsBase.hasNextTest()) {
            TestsBase.Test currentTest = testsBase.nextTest();
            int i = testsBase.getCurrentTestIndex();

            weightsErrorGradients[i] = new WeightsOutputErrorGradient(network, currentTest).getOutputErrorGradient();
            biasesErrorGradients[i] = new BiasesOutputErrorGradient(network, currentTest).getOutputErrorGradient();
        }

        // Calculating sum of weights error gradients
        double[] averageWeightsErrorGradient = weightsErrorGradients[0];

        for (int i = 1; i < weightsErrorGradients.length; ++i) {
            for (int j = 0; j < weightsErrorGradients[i].length; ++j) {
                averageWeightsErrorGradient[j] += weightsErrorGradients[i][j];
            }
        }
        
        // Calculating sum of biases error gradients
        double[] averageBiasesErrorGradient = biasesErrorGradients[0];

        for (int i = 1; i < biasesErrorGradients.length; ++i) {
            for (int j = 0; j < biasesErrorGradients[i].length; ++j) {
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

    // Trains network by tweaking its parameters after any test's error calculated
    public NeuralNetwork trainNetworkOnline(int trainEpochsQuantity, double maxAcceptableAverageOutputError) {
        boolean isNetworkTrainedEnough = false;

        do {
            double averageOutputError = 0;
            lastTrainErrorsData = new double[trainEpochsQuantity];

            for (int i = 0; i < trainEpochsQuantity; ++i) {
                network = trainEpochOnline(new NeuralNetwork(network));

                // Average error calculating
                testsBase.clearTestsQueue();

                do {
                    TestsBase.Test currentTest = testsBase.nextTest();
                    averageOutputError += calcOutputError(network.calcOutputBy(currentTest.input()), currentTest.correctOutput());
                } while (testsBase.hasNextTest());

                averageOutputError /= testsBase.size;
                lastTrainErrorsData[i] = averageOutputError;
            }

            System.out.println(averageOutputError);

            // If network wasn't trained enough to have average error smaller than maximal acceptable error - training restarts
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

        testsBase.clearTestsQueue();

        while (testsBase.hasNextTest()) {
            TestsBase.Test currentTest = testsBase.nextTest();
            currentWeightGradient = new WeightsOutputErrorGradient(network, currentTest).getOutputErrorGradient();
            currentBiasGradient = new BiasesOutputErrorGradient(network, currentTest).getOutputErrorGradient();

            tweakNetworkParametersByGradients(network, currentWeightGradient, currentBiasGradient);
        }

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

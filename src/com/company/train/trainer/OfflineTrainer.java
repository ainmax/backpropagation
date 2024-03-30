package com.company.train.trainer;

import com.company.model.Matrix;
import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;
import com.company.train.gradient.BiasesOutputErrorGradient;
import com.company.train.gradient.WeightsOutputErrorGradient;

public class OfflineTrainer extends Trainer {
    public OfflineTrainer(NeuralNetwork network, TestSet testSet, TrainerOptions options) {
        super(network, testSet, options);
    }

    // Trains network by tweaking its parameters after all tests' errors calculated
    public NeuralNetwork trainNetwork() {
        boolean isNetworkTrainedEnough = false;

        do {
            lastTrainErrorsData = new double[options.trainEpochsCount()];

            // Define zero increments
            previousWeightsIncrements = new Matrix[network.weights.length];
            previousBiasesIncrements = new Matrix[network.biases.length];

            for (int i = 0; i < network.weights.length; ++i) {
                previousWeightsIncrements[i] = new Matrix(network.weights[i].N, network.weights[i].M);
            }

            for (int i = 0; i < network.biases.length; ++i) {
                previousBiasesIncrements[i] = new Matrix(network.biases[i].N, network.biases[i].M);
            }

            for (int i = 0; i < options.trainEpochsCount(); ++i) {
                // Save old parameters
                Matrix[] oldWeights = new Matrix[network.weights.length];
                Matrix[] oldBiases = new Matrix[network.biases.length];

                for (int p = 0; p < network.weights.length; ++p) {
                    oldWeights[p] = new Matrix(network.weights[p]);
                }

                for (int p = 0; p < network.biases.length; ++p) {
                    oldBiases[p] = new Matrix(network.biases[p]);
                }

                // Tweak network's parameters
                network = trainEpoch(new NeuralNetwork(network));

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

                // Average error calculating
                testSet.clearTestsQueue();

                double maxOutputError = 0;
                double averageOutputError = 0;

                do {
                    TestSet.Test currentTest = testSet.nextTest();

                    averageOutputError += calcOutputError(network.calcOutputBy(currentTest.input()), currentTest.correctOutput());
                    maxOutputError = Math.max(maxOutputError, calcOutputError(network.calcOutputBy(currentTest.input()), currentTest.correctOutput()));
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

    private NeuralNetwork trainEpoch(NeuralNetwork network) {
        double[][] weightsErrorGradients = new double[testSet.size][];
        double[][] biasesErrorGradients = new double[testSet.size][];

        // Define gradient for each test
        testSet.clearTestsQueue();

        while (testSet.hasNextTest()) {
            TestSet.Test currentTest = testSet.nextTest();
            int i = testSet.getCurrentTestIndex();

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
            averageWeightsErrorGradient[i] /= testSet.size;
        }

        for (int i = 0; i < averageBiasesErrorGradient.length; ++i) {
            averageBiasesErrorGradient[i] /= testSet.size;
        }

        return tweakNetworkParametersByGradients(network, averageWeightsErrorGradient, averageBiasesErrorGradient);
    }
}

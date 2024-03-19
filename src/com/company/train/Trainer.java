package com.company.train;

import com.company.model.Matrix;
import com.company.model.NeuralNetwork;

public class Trainer {
    private NeuralNetwork network;
    private final TestsBase testsBase;

    // Gradients by network output
    private final Matrix[] weightsErrorGradients;
    private final Matrix[] biasesErrorGradients;

    private static final double LEARN_SPEED = 0.1;
    private static final double COMPRESSION_COEFFICIENT = 1;
//    private static final double INERTIAL_COEFFICIENT = 0.1;

    public Trainer(NeuralNetwork neuralNetwork, TestsBase testsBase) {
        network = neuralNetwork;
        this.testsBase = testsBase;
        weightsErrorGradients = new Matrix[testsBase.size];
        biasesErrorGradients = new Matrix[testsBase.size];
    }

    public NeuralNetwork getNetwork() {
        return network;
    }

    // Returns average error after every epoch
    public double[] trainNetworkOffline(int epochsQuantity, double maxAcceptableError) {
        double[] errors = new double[epochsQuantity];

        for (int i = 0; i < epochsQuantity; ++i) {
            errors[i] = trainEpochOffline();

            if (i == epochsQuantity - 1 && errors[i] > maxAcceptableError) {
                System.out.println(errors[i]);
                i = -1;
                network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
            }
        }

        return errors;
    }

    // Returns average error after every epoch
    public double[] trainNetworkOnline(int epochsQuantity, double maxAcceptableError) {
        double[] errors = new double[epochsQuantity];

        for (int i = 0; i < epochsQuantity; ++i) {
            errors[i] = trainEpochOnline();

            if (i == epochsQuantity - 1 && errors[i] > maxAcceptableError) {
                System.out.println(errors[i]);
                i = -1;
                network = new NeuralNetwork(network.inputSize, network.outputSize, network.hiddenLayersSizes);
            }
        }

        return errors;
    }

    private static double calcOutputError(Matrix output, double[] expectedOutput) {
        double outputError = 0;

        for (int i = 0; i < output.N; ++i) {
            outputError += (output.values[i][0] - expectedOutput[i]) * (output.values[i][0] - expectedOutput[i]);
        }

        return outputError;
    }

    // Reweight network by gradients and returns average error
    private double trainEpochOffline() {
        int i = 0;

        // Calculating gradient for each test
        while (testsBase.hasNextTest()) {
            calcErrorGradients(testsBase.nextTest(), i);
            
            ++i;
        }

        testsBase.clearTestsQueue();
        
        Matrix averageWeightsErrorGradient = new Matrix(weightsErrorGradients[0]);
        Matrix averageBiasesErrorGradient = new Matrix(biasesErrorGradients[0]);
        
        for (i = 1; i < weightsErrorGradients.length; ++i) {
            averageWeightsErrorGradient = averageWeightsErrorGradient.plus(weightsErrorGradients[i]);
        }

        for (i = 1; i < biasesErrorGradients.length; ++i) {
            averageBiasesErrorGradient = averageBiasesErrorGradient.plus(biasesErrorGradients[i]);
        }

        // Average gradient by all tests
        averageWeightsErrorGradient = averageWeightsErrorGradient.multiply( (double)1 / testsBase.size);
        averageBiasesErrorGradient = averageBiasesErrorGradient.multiply((double)1 / testsBase.size);

        reWeightNetwork(averageWeightsErrorGradient, averageBiasesErrorGradient);

        // Average input error on this reweight
        double averageOutputError = 0;
        TestsBase.Test test;

        while (testsBase.hasNextTest()) {
            test = testsBase.nextTest();
            network.setInputLayer(test.input());
            network.generateOutputLayer();

            averageOutputError += calcOutputError(network.getOutputLayer(), test.correctOutput());
        }

        averageOutputError /= testsBase.size;

        testsBase.clearTestsQueue();

        return averageOutputError;
    }

    private double trainEpochOnline() {
        int i = 0;

        // Calculating gradient for each test
        while (testsBase.hasNextTest()) {
            calcErrorGradients(testsBase.nextTest(), i);
            reWeightNetwork(weightsErrorGradients[i], biasesErrorGradients[i]);

            ++i;
        }

        testsBase.clearTestsQueue();

        // Average input error on this reweight
        double averageOutputError = 0;
        TestsBase.Test test;

        while (testsBase.hasNextTest()) {
            test = testsBase.nextTest();
            network.setInputLayer(test.input());
            network.generateOutputLayer();

            averageOutputError += calcOutputError(network.getOutputLayer(), test.correctOutput());
        }

        averageOutputError /= testsBase.size;

        testsBase.clearTestsQueue();

        return averageOutputError;
    }

    private void calcErrorGradients(TestsBase.Test test, int iterationIndex) {
        network.setInputLayer(test.input());
        network.generateOutputLayer();

        Matrix weightsGradient = new Matrix(network.getWeightGradient());
        Matrix biasesGradient = new Matrix(network.getBiasesGradient());

        // Making from "output gradient" - "error gradient"
        for (int j = 0; j < weightsGradient.N; ++j) {
            for (int k = 0; k < weightsGradient.M; k++) {
                weightsGradient.values[j][k] *= 2 * (network.getOutputLayer().values[j][0] - test.correctOutput()[j]);
            }
        }

        // Making from "output gradient" - "error gradient"
        for (int j = 0; j < biasesGradient.N; ++j) {
            for (int k = 0; k < biasesGradient.M; k++) {
                biasesGradient.values[j][k] *= 2 * (network.getOutputLayer().values[j][0] - test.correctOutput()[j]);
            }
        }

        Matrix weightsErrorGradient = new Matrix(1, weightsGradient.M);
        Matrix biasesErrorGradient = new Matrix(1, biasesGradient.M);

        for (int j = 0; j < weightsGradient.M; ++j) {
            for (int k = 0; k < weightsGradient.N; k++) {
                weightsErrorGradient.values[0][j] += weightsGradient.values[k][j];
            }
        }

        for (int j = 0; j < biasesGradient.M; ++j) {
            for (int k = 0; k < biasesGradient.N; k++) {
                biasesErrorGradient.values[0][j] += biasesGradient.values[k][j];
            }
        }

        weightsErrorGradients[iterationIndex] = weightsErrorGradient;
        biasesErrorGradients[iterationIndex] = biasesErrorGradient;
    }

    private void reWeightNetwork(Matrix weightsErrorGradient, Matrix biasesErrorGradient/*, Matrix weightInertia, Matrix biasInertia*/) {
        int currentWeightIndex = 0;

        // Weights reweight
        for (int i = 0; i < network.weights.length; ++i) {
            for (int j = 0; j < network.weights[i].N; ++j) {
                for (int k = 0; k < network.weights[i].M; ++k, ++currentWeightIndex) {
                    network.weights[i].values[j][k] = COMPRESSION_COEFFICIENT * network.weights[i].values[j][k] - LEARN_SPEED * weightsErrorGradient.values[0][currentWeightIndex]/* - INERTIAL_COEFFICIENT * weightInertia.values[j][k]*/;
                }
            }
        }

        int currentBiasIndex = 0;

        // Biases reweight
        for (int i = 0; i < network.biases.length; ++i) {
            for (int j = 0; j < network.biases[i].N; ++j, ++currentBiasIndex) {
                network.biases[i].values[j][0] = COMPRESSION_COEFFICIENT * network.biases[i].values[j][0] - LEARN_SPEED * biasesErrorGradient.values[0][currentBiasIndex]/* - INERTIAL_COEFFICIENT * biasInertia.values[0][currentBiasIndex]*/;
            }
        }
    }
}

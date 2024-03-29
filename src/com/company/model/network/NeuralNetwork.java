package com.company.model.network;

import com.company.model.Matrix;
import com.company.model.RandomMatrix;

// Neural network model
public class NeuralNetwork {
    // Options
    public final int inputSize;
    public final int outputSize;
    public final int hiddenLayersCount;
    public final int[] hiddenLayersSizes;

    // Structure elements
    public Layer outputLayer;
    public Layer[] hiddenLayers;

    // Current parameters values
    public Matrix[] weights;
    public Matrix[] biases;

    // Throws exception if hiddenLayers is empty
    public NeuralNetwork(int inputSize, int outputSize, int[] hiddenLayersSizes) {
        hiddenLayersCount = hiddenLayersSizes.length;

        if (hiddenLayersCount == 0) {
            throw new IllegalArgumentException("There is no way to create network without hidden layers.");
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayersSizes = hiddenLayersSizes.clone();

        defineBlankStructureByCurrentOptions();
        fillParametersWithRandomValues();
    }

    // Creates network by another network's options and parameters
    public NeuralNetwork(NeuralNetwork network) {
        inputSize = network.inputSize;
        outputSize = network.outputSize;
        hiddenLayersCount = network.hiddenLayersCount;
        hiddenLayersSizes = network.hiddenLayersSizes.clone();

        defineBlankStructureByCurrentOptions();
        fillParametersWithRandomValues();

        for (int i = 0; i < weights.length; ++i) {
            weights[i] = new Matrix(network.weights[i]);
        }

        for (int i = 0; i < biases.length; ++i) {
            biases[i] = new Matrix(network.biases[i]);
        }
    }

    private void defineBlankStructureByCurrentOptions() {
        hiddenLayers = new Layer[hiddenLayersCount];

        // Define layers
        for (int i = 0; i < hiddenLayersCount; ++i) {
            hiddenLayers[i] = new Layer();
        }

        outputLayer = new Layer();
    }

    private void fillParametersWithRandomValues() {
        biases = new Matrix[hiddenLayersCount + 1];
        weights = new Matrix[hiddenLayersCount + 1];

        // Fill parameters
        weights[0] = new RandomMatrix(hiddenLayersSizes[0], inputSize);

        for (int i = 0; i < hiddenLayersCount; ++i) {
            biases[i] = new RandomMatrix(hiddenLayersSizes[i], 1);

            if (i < hiddenLayersCount - 1) {
                weights[i + 1] = new RandomMatrix(hiddenLayersSizes[i + 1], hiddenLayersSizes[i]);
            }
        }

        biases[hiddenLayersCount] = new RandomMatrix(outputSize, 1);
        weights[hiddenLayersCount] = new RandomMatrix(outputSize, hiddenLayersSizes[hiddenLayersCount - 1]);
    }

    public double[] calcOutputBy(double[] inputValues) {
        Matrix previousLayerOutput = new Matrix(inputSize, 1, inputValues);

        for (int i = 0; i < hiddenLayersCount; ++i) {
            // Formula for layer output is W * A + B, where W - weights between current and next layers, B - biases
            previousLayerOutput = hiddenLayers[i].apply(weights[i].multiply(previousLayerOutput).plus(biases[i]));
        }

        // Formula for layer output is W * A + B, where W - weights between current and next layers, B - biases
        Matrix outputMatrix = outputLayer.apply(weights[hiddenLayersCount].multiply(previousLayerOutput).plus(biases[hiddenLayersCount]));

        // Convert matrix to array
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; ++i) {
            output[i] = outputMatrix.values[i][0];
        }

        return output;
    }

    public class Layer { /// todo rename to HiddenLayer

        private Layer() {}

        public Matrix apply(Matrix previousLayerOutput) {
            Matrix activatedValues = new Matrix(previousLayerOutput);

            // Each element in nodesValues transform by activate function
            for (int i = 0; i < previousLayerOutput.N; ++i) {
                activatedValues.values[i][0] = calcActivatedNodeValue(previousLayerOutput.values[i][0]);
            }

            return activatedValues;
        }

        // Uses sigmoid function
        private double calcActivatedNodeValue(double nodeCharge) {
            return 1 / (1 + Math.pow(Math.exp(1), -nodeCharge));
        }
    }
}

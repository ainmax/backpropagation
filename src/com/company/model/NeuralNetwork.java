package com.company.model;

// Neural network model
public class NeuralNetwork {
    // Options
    public final int inputSize;
    public final int outputSize;
    public final int hiddenLayersCount;
    public final int[] hiddenLayersSizes;

    // Structure elements
    Layer outputLayer;
    Layer[] hiddenLayers;

    // Current parameters values
    Matrix[] weights;
    Matrix[] biases;

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

    private void defineBlankStructureByCurrentOptions() {
        hiddenLayers = new Layer[hiddenLayersCount];
        outputLayer = new Layer(outputSize, hiddenLayersCount);

        // Define layers
        for (int i = 0; i < hiddenLayersCount; ++i) {
            hiddenLayers[i] = new Layer(hiddenLayersSizes[i], i);
        }
    }

    private void fillParametersWithRandomValues() {
        biases = new Matrix[hiddenLayersCount + 1];
        weights = new Matrix[hiddenLayersCount + 1];
        weights[0] = new RandomMatrix(hiddenLayersSizes[0], inputSize);

        for (int i = 0; i < hiddenLayersCount; ++i) {
            biases[i] = new RandomMatrix(hiddenLayersSizes[i], 1);

            if (i < hiddenLayersCount - 1) {
                weights[i + 1] = new Matrix(hiddenLayersSizes[i + 1], hiddenLayersSizes[i]);
            }
        }

        biases[hiddenLayersCount] = new RandomMatrix(outputSize, 1);
        weights[hiddenLayersCount] = new RandomMatrix(outputSize, hiddenLayersSizes[hiddenLayersCount - 1]);
    }

    public double[] calcOutputBy(double[] inputValues) {
        Matrix inputLayer = new Matrix(inputSize, 1, inputValues);
        Matrix previousLayerOutput = weights[0].multiply(inputLayer).plus(biases[0]);

        for (int i = 0; i < hiddenLayersCount; ++i) {
            previousLayerOutput = hiddenLayers[i].calcOutputBy(previousLayerOutput);
        }

        Matrix outputMatrix = outputLayer.calcActivatedValues(previousLayerOutput);
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; ++i) {
            output[i] = outputMatrix.values[i][0];

        }

        return output;
    }

    class Layer {
        final int index;
        final int size;

        Layer(int size, int index) {
            this.index = index;
            this.size = size;
        }

        Matrix calcOutputBy(Matrix previousLayerOutput) {
            Matrix activatedNodesValues = calcActivatedValues(previousLayerOutput);

            // Formula for layer output is W * A + B, where W - weights between current and next layers, B - biases
            return weights[index + 1].multiply(activatedNodesValues).plus(biases[index + 1]);
        }

        Matrix calcActivatedValues(Matrix previousLayerOutput) {
            Matrix activatedValues = new Matrix(previousLayerOutput);

            // Each element in nodesValues transform by activate function
            for (int i = 0; i < size; ++i) {
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

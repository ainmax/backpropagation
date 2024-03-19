package com.company.model;

// Neural network model
public class NeuralNetwork {
    // Options
    public final int inputSize;
    public final int outputSize;
    public final int hiddenLayersCount;
    public final int[] hiddenLayersSizes;

    // Structure elements
    private Matrix inputLayer;
    private Layer outputLayer;
    private Layer[] hiddenLayers;

    // Current parameters values
    private Matrix[] weights;
    private Matrix[] biases;

    private WeightsGradient weightsGradient;
    private BiasesGradient biasesGradient;

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
    }

    private void defineBlankStructureByCurrentOptions() {
        inputLayer = new Matrix(inputSize, 1);
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

        // Creating empty matrices and layers
        for (int i = 0; i < hiddenLayersCount; ++i) {
            biases[i] = new RandomMatrix(hiddenLayersSizes[i], 1);

            if (i < hiddenLayersCount - 1) {
                weights[i + 1] = new Matrix(hiddenLayersSizes[i + 1], hiddenLayersSizes[i]);
            }
        }

        biases[hiddenLayersCount] = new RandomMatrix(outputSize, 1);
        weights[hiddenLayersCount] = new RandomMatrix(outputSize, hiddenLayersSizes[hiddenLayersCount - 1]);
    }

    public void setInputLayer(double[] inputLayer) {
        this.inputLayer = new Matrix(inputLayer.length, 1, inputLayer);
    }

    public Matrix getOutputLayer() {
        if (inputLayer.N == 0) {
            throw new RuntimeException("Input haven't been set yet");
        }

        return outputLayer.activatedNodesValues;
    }

    public void generateOutputLayer() {
        if (inputLayer.N == 0) {
            throw new RuntimeException("Input haven't been set yet");
        }

        // InputLayer output
        Matrix previousLayerOutput = weights[0].multiply(inputLayer).plus(biases[0]);

        for (int i = 0; i < hiddenLayersCount; ++i) {
            hiddenLayers[i].activate(previousLayerOutput);
            previousLayerOutput = hiddenLayers[i].generateOutput();
        }

        outputLayer.activate(previousLayerOutput);
    }

    public Matrix getWeightGradient() {
        weightsGradient.calc();
        return weightsGradient.getGradient();
    }

    public Matrix getBiasesGradient() {
        biasesGradient.calc();
        return biasesGradient.getGradient();
    }

    // Uses sigmoid function
    static double calcActivatedNodeValue(double nodeCharge) {
        return 1 / (1 + Math.pow(Math.exp(1), -nodeCharge));
    }

    // Sigmoid function first derivative
    static double calcActivatedNodeValueDerivative(double nodeCharge) {
        return 1 / (2 + Math.pow(Math.exp(1), -nodeCharge) + Math.pow(Math.exp(1), nodeCharge));
    }

    private class Layer {
        final int index;
        final int size;
        Matrix nodesValues;
        Matrix activatedNodesValues;

        Layer(int size, int index) {
            this.index = index;
            this.size = size;
            nodesValues = new Matrix(size, 1);
            activatedNodesValues = new Matrix(size, 1);
        }

        // Each element in nodesValues transform to calcActivatedNodeValue(this element)
        void activate(Matrix in) {
            nodesValues = new Matrix(in);

            for (int i = 0; i < activatedNodesValues.N; ++i) {
                for (int j = 0; j < activatedNodesValues.M; ++j) {
                    activatedNodesValues.values[i][j] = calcActivatedNodeValue(nodesValues.values[i][j]);
                }
            }
        }

        Matrix generateOutput() {
            return weights[index + 1].multiply(activatedNodesValues).plus(biases[index + 1]);
        }
    }

    private class WeightsGradient {
        Layer currentLayer;
        Layer previousLayer;

        Matrix[] currentGradient;

        /*
        Gradient storage structure:

                  W(0)     W(1)     W(2)    ...     W(n)
          A(0)  G(0)(0)     -        -      ...      -
          A(1)  G(1)(0)  G(1)(1)     -      ...      -
          A(2)  G(2)(0)  G(2)(1)  G(1)(2)   ...      -
            .      .        .        .       .       .
            .      .        .        .       .       .
            .      .        .        .       .       .
          A(n)  G(n)(0)  G(n)(1)  G(n)(2)   ...   G(n)(n)

          Where n = hiddenLayersQuantity,
          A(i) - nodes activations vector,
          W(j) - weights vector (weights matrices transform to vectors by indexing element (k, r) as [k * M + r], where M - matrix v-dimension),
          G(i)(j) = Gij - gradient matrix:

                                W(j)[0]         W(j)[1]         W(j)[2]      ...      W(j)[N * M - 1]
                 A(i)[0]       Gij[0][0]       Gij[0][1]       Gij[0][2]     ...     Gij[0][N * M - 1]
                 A(i)[1]       Gij[1][0]       Gij[1][1]       Gij[1][2]     ...     Gij[1][N * M - 1]
                 A(i)[2]       Gij[2][0]       Gij[2][1]       Gij[2][2]     ...     Gij[2][N * M - 1]
                    .              .               .               .          .              .
                    .              .               .               .          .              .
                    .              .               .               .          .              .
               A(i)[S - 1]   Gij[S - 1][0]   Gij[S - 1][1]   Gij[S - 1][2]   ...   Gij[S - 1][N * M - 1]

               Where N - W(j) v-dimension,
               M - W(j) - h-dimension,
               Gij[u][v] - partial derivative of A(i)[u] by W(j)[v]
         */

        WeightsGradient() {

        }

        // Returns gradient of outputLayer
        public Matrix getGradient() {
            if (currentGradient.length == 0) {
                throw new RuntimeException("Gradient wasn't calculated.");
            }

            return currentGradient[hiddenLayersCount];
        }

        // Calculates base of recursion - partial derivatives of first hiddenLayer activations by parameters, connected to inputLayer
        public void calc() {
            currentLayer = hiddenLayers[0];
            currentGradient = new Matrix[hiddenLayersCount + 1];
            currentGradient[0] = new Matrix(currentLayer.size, inputSize * currentLayer.size);

            for (int i = 0; i < currentLayer.size; ++i) {
                for (int k = 0; k < inputSize; ++k) {
                    currentGradient[0].values[i][i * inputSize + k] = inputLayer.values[k][0];
                    // Adds activation function partial derivative factor
                    currentGradient[0].values[i][i * inputSize + k] *= calcActivatedNodeValueDerivative(currentLayer.nodesValues.values[i][0]);
                }
            }

            previousLayer = currentLayer;
            calcNextLayer();
        }

        // Recursive function calculates all partial derivatives of next layer activations by all previous parameters, using partial derivatives which were calculated earlier
        private void calcNextLayer() {
            if (currentLayer.index + 1 > hiddenLayersCount) {
                return;
            }

            if (currentLayer.index + 1 == hiddenLayersCount) {
                currentLayer = outputLayer;
            } else {
                currentLayer = hiddenLayers[currentLayer.index + 1];
            }

            currentGradient[currentLayer.index] = new Matrix(currentLayer.size, currentGradient[previousLayer.index].M + currentLayer.size * previousLayer.size);

            for (int i = 0; i < currentLayer.size; ++i) {
                for (int k = 0; k < previousLayer.size; ++k) {
                    currentGradient[currentLayer.index].values[i][i * previousLayer.size + k + currentGradient[previousLayer.index].M] = previousLayer.nodesValues.values[k][0];
                    // Adds activation function partial derivative factor
                    currentGradient[currentLayer.index].values[i][i * previousLayer.size + k + currentGradient[previousLayer.index].M] *= calcActivatedNodeValueDerivative(currentLayer.nodesValues.values[i][0]);
                }
            }

            // Partial derivative by parameters from previous layers (not by weights or biases, connected to current layer)
            Matrix deepGradient = weights[currentLayer.index].multiply(currentGradient[previousLayer.index]);

            for (int i = 0; i < currentLayer.size; ++i) {
                for (int j = 0; j < currentGradient[previousLayer.index].M; ++j) {
                    currentGradient[currentLayer.index].values[i][j] = deepGradient.values[i][j];
                    // Adds activation function partial derivative factor
                    currentGradient[currentLayer.index].values[i][j] *= calcActivatedNodeValueDerivative(currentLayer.nodesValues.values[i][0]);
                }
            }

            previousLayer = currentLayer;
            calcNextLayer();
        }
    }

    // Almost same, as WeightsGradient class. Only difference is: derivative by bias on the previous layer - 1. All class logic described in WeightsGradient class.
    private class BiasesGradient {
        Layer currentLayer;
        Layer previousLayer;

        Matrix[] currentGradient;

        BiasesGradient() {

        }

        public Matrix getGradient() {
            if (currentGradient.length == 0) {
                throw new RuntimeException("Gradient wasn't calculated.");
            }

            return currentGradient[hiddenLayersCount];
        }

        public void calc() {
            currentLayer = hiddenLayers[0];
            currentGradient = new Matrix[hiddenLayersCount + 1];
            currentGradient[0] = new Matrix(currentLayer.size, inputSize * currentLayer.size);

            for (int i = 0; i < currentLayer.size; ++i) {
                for (int k = 0; k < inputSize; ++k) {
                    currentGradient[0].values[i][i * inputSize + k] = 1;
                    currentGradient[0].values[i][i * inputSize + k] *= calcActivatedNodeValueDerivative(currentLayer.nodesValues.values[i][0]);
                }
            }

            previousLayer = currentLayer;
            calcNextLayer();
        }

        public void calcNextLayer() {
            if (currentLayer.index + 1 > hiddenLayersCount) {
                return;
            }

            if (currentLayer.index + 1 == hiddenLayersCount) {
                currentLayer = outputLayer;
            } else {
                currentLayer = hiddenLayers[currentLayer.index + 1];
            }

            currentGradient[currentLayer.index] = new Matrix(currentLayer.size, currentGradient[previousLayer.index].M + currentLayer.size * previousLayer.size);

            for (int i = 0; i < currentLayer.size; ++i) {
                for (int k = 0; k < previousLayer.size; ++k) {
                    currentGradient[currentLayer.index].values[i][i * previousLayer.size + k + currentGradient[previousLayer.index].M] = 1;
                    currentGradient[currentLayer.index].values[i][i * previousLayer.size + k + currentGradient[previousLayer.index].M] *= calcActivatedNodeValueDerivative(currentLayer.nodesValues.values[i][0]);
                }
            }

            Matrix deepGradient = weights[currentLayer.index].multiply(currentGradient[previousLayer.index]);

            for (int i = 0; i < currentLayer.size; ++i) {
                for (int j = 0; j < currentGradient[previousLayer.index].M; ++j) {
                    currentGradient[currentLayer.index].values[i][j] = deepGradient.values[i][j];
                    currentGradient[currentLayer.index].values[i][j] *= calcActivatedNodeValueDerivative(currentLayer.nodesValues.values[i][0]);
                }
            }

            previousLayer = currentLayer;
            calcNextLayer();
        }
    }
}

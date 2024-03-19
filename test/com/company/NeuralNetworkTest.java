package com.company;

import com.company.model.Matrix;
import com.company.model.NeuralNetwork;
import org.junit.jupiter.api.Test;

class NeuralNetworkTest {

    @Test
    void checkNeuralNetworkAnswer() {
        NeuralNetwork neuralNetwork = new NeuralNetwork(1, 1, new int[] {1});

        double[] input = {1};

        neuralNetwork.setInputLayer(input);
        neuralNetwork.generateOutputLayer();
        Matrix output = neuralNetwork.getOutputLayer();
        System.out.println(output);
    }
}
package com.company.model;

import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;

public class BiasesOutputErrorGradient extends NetworkOutputErrorGradient {
    public BiasesOutputErrorGradient(NeuralNetwork network, TestSet.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double linearCoefficient) {
        return 1;
    }

    int getConnectedParametersCount(int previousLayerSize, int currentLayerSize) {
        return currentLayerSize;
    }
}

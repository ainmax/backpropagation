package com.company.model;

import com.company.train.TestsBase;

public class WeightsOutputErrorGradient extends NetworkOutputErrorGradient {
    public WeightsOutputErrorGradient(NeuralNetwork network, TestsBase.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double linearCoefficient) {
        return linearCoefficient;
    }

    int getConnectedParametersCount(int previousLayerSize, int currentLayerSize) {
        return previousLayerSize * currentLayerSize;
    }
}

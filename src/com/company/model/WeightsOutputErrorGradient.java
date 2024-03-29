package com.company.model;

import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;

public class WeightsOutputErrorGradient extends NetworkOutputErrorGradient {
    public WeightsOutputErrorGradient(NeuralNetwork network, TestSet.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double linearCoefficient) {
        return linearCoefficient;
    }

    int getConnectedParametersCount(int previousLayerSize, int currentLayerSize) {
        return previousLayerSize * currentLayerSize;
    }
}

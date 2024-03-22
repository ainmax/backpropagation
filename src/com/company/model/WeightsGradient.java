package com.company.model;

import com.company.train.TestsBase;

public class WeightsGradient extends NetworkGradient {
    WeightsGradient(NeuralNetwork network, double[] inputValues, TestsBase.Test test) {
        super(network, inputValues, test);
    }

    double calcLinearFirstDerivative(double lineCoefficient) {
        return lineCoefficient;
    }
}

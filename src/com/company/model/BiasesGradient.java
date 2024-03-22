package com.company.model;

import com.company.train.TestsBase;

public class BiasesGradient extends NetworkGradient {
    BiasesGradient(NeuralNetwork network, double[] inputValues, TestsBase.Test test) {
        super(network, inputValues, test);
    }

    double calcLinearFirstDerivative(double lineCoefficient) {
        return 1;
    }
}

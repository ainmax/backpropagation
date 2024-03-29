package com.company.train;

import java.util.Arrays;
import java.util.function.Function;

public enum TestFunctionsEnum {
    SIMPLE(s -> s),
    ONE_QUANTITY(s -> {
        double[] output = new double[s.length + 1];
        output[(int) Arrays.stream(s).filter(i -> i == 1).count()] = 1;

        return output;
    }),
    PROJECTION(s -> new double[] { s[0] }),
    DIGIT_RECOGNIZE(s -> {
        double[] output = new double[10];

        output[0] = 1;

        return output;
    });

    public final Function<double[], double[]> answerFunction;

    TestFunctionsEnum(Function<double[], double[]> answerFunction) {
        this.answerFunction = answerFunction;
    }
}

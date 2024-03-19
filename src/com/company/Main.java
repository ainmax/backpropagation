package com.company;

import com.company.model.NeuralNetwork;
import com.company.train.TestFunctionsEnum;
import com.company.train.TestsBase;
import com.company.train.Trainer;

import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 3, new int[] {3});
        TestsBase testsBase = new TestsBase(neuralNetwork.inputSize, TestFunctionsEnum.ONE_QUANTITY.answerFunction);
        Trainer trainer = new Trainer(neuralNetwork, testsBase);

        // Average error after every weights change
        double[] errors = trainer.trainNetworkOffline(10000, 0.1);
        neuralNetwork = trainer.getNetwork();

        for (int i = 0; i < neuralNetwork.weights.length; ++i) {
            System.out.println(neuralNetwork.weights[i]);
            System.out.println();
        }

        for (int i = 0; i < neuralNetwork.biases.length; ++i) {
            System.out.println(neuralNetwork.biases[i]);
            System.out.println();
        }

        System.out.println(Arrays.toString(errors));
        System.out.println("--------------------------------------");

        Scanner in = new Scanner(System.in);
        double[] userInput = new double[neuralNetwork.inputSize];

        while (true) {
            try {
                for (int i = 0; i < neuralNetwork.inputSize; ++i) {
                    userInput[i] = in.nextDouble();
                }
            } catch (Exception e) {
                System.out.println();
                System.out.println("--------------------------------------");
                break;
            }

            neuralNetwork.setInputLayer(userInput);
            neuralNetwork.generateOutputLayer();
            System.out.println();
            System.out.print("Output: ");
            System.out.println(neuralNetwork.getOutputLayer());
            System.out.print("Correct output: ");
            System.out.println(Arrays.toString(TestFunctionsEnum.ONE_QUANTITY.answerFunction.apply(userInput)));
            System.out.println("--------------------------------------");
        }

        in.close();
    }
}

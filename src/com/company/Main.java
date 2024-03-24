package com.company;

import com.company.model.NeuralNetwork;
import com.company.train.TestFunctionsEnum;
import com.company.train.TestsBase;
import com.company.train.Trainer;

import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        TestFunctionsEnum testFunction = TestFunctionsEnum.ONE_QUANTITY;
        NeuralNetwork neuralNetwork = new NeuralNetwork(4, 5, new int[] {3, 3});
        TestsBase testsBase = new TestsBase(neuralNetwork.inputSize, testFunction.answerFunction);

        Trainer trainer = new Trainer(neuralNetwork, testsBase);
        neuralNetwork = trainer.trainNetworkOffline(10000, 0.01);

        double[] neuralNetworkErrors = trainer.getLastTrainErrorsData();

        for (int i = 0; i < neuralNetwork.weights.length; ++i) {
            System.out.println(neuralNetwork.weights[i]);
            System.out.println();
        }

        for (int i = 0; i < neuralNetwork.biases.length; ++i) {
            System.out.println(neuralNetwork.biases[i]);
            System.out.println();
        }

        System.out.println(Arrays.toString(neuralNetworkErrors));
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

            System.out.println();
            System.out.print("Output: ");
            System.out.println(Arrays.toString(neuralNetwork.calcOutputBy(userInput)));
            System.out.print("Correct output: ");
            System.out.println(Arrays.toString(testFunction.answerFunction.apply(userInput)));
            System.out.println("--------------------------------------");
        }

        in.close();
    }
}

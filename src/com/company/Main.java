package com.company;

import com.company.model.Matrix;
import com.company.model.network.NeuralNetwork;
import com.company.train.TestFunctionsEnum;
import com.company.train.TestSet;
import com.company.train.trainer.OfflineTrainer;
import com.company.train.trainer.OnlineTrainer;
import com.company.train.trainer.Trainer;
import com.company.train.trainer.TrainerOptions;

import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        TestFunctionsEnum testFunction = TestFunctionsEnum.ONE_QUANTITY;
        NeuralNetwork neuralNetwork = new NeuralNetwork(10, 11, new int[] {8, 8});
        TrainerOptions trainerOptions = new TrainerOptions(0.4, 0.2, 1000, 50, 0.02, 3);

        TestSet testSet = new TestSet(neuralNetwork.inputSize, testFunction.answerFunction);
        Trainer offlineTrainer = new OfflineTrainer(neuralNetwork, testSet, trainerOptions);

        neuralNetwork = offlineTrainer.trainNetwork();

        double[] neuralNetworkErrors = offlineTrainer.getLastTrainErrorsData();
        System.out.println(Arrays.toString(neuralNetworkErrors));

        for (int i = 0; i < neuralNetwork.weights.length; ++i) {
            System.out.println(neuralNetwork.weights[i]);
            System.out.println();
        }

        for (int i = 0; i < neuralNetwork.biases.length; ++i) {
            System.out.println(neuralNetwork.biases[i]);
            System.out.println();
        }

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
            System.out.print("Precise output: ");
            System.out.println(Arrays.toString(neuralNetwork.calcOutputBy(userInput)));
            System.out.print("Simplified output: ");
            System.out.println(new Matrix(1, neuralNetwork.outputSize, neuralNetwork.calcOutputBy(userInput)));
            System.out.print("Correct output: ");
            System.out.println(new Matrix(1, neuralNetwork.outputSize, testFunction.answerFunction.apply(userInput)));
            System.out.println("--------------------------------------");
        }

        in.close();
    }
}

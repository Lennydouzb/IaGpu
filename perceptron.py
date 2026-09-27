import pandas as pd
from matrix_lib import MatrixFloat, e_types
import argparse
import math
import random
import matplotlib.pyplot as plt
pd.set_option("future.no_silent_downcasting", True)
def relu(data):
    tmp = []
    for i in data:
        if i < 0:
            tmp.append(0)
        else:
            tmp.append(i)
    return tmp
def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "--layers",
                type = int,
                nargs="+",
                default= [5, 5],
                help="List of layer sizes (ex: 5 5 6, is 2 layers of 5 neurons and one of 6, then output)"
                )
        parser.add_argument(
                "--epochs",
                type = int,
                default= 100,
                help="How much epochs you want the learning to be"
                )
        parser.add_argument(
                "--learning-rate",
                type = float,
                default= 0.1,
                help="How fast you want it to learn, the more its fast, the less its precise"
                )
        parser.add_argument(
                "--parquet",
                type = str,
                required = True,
                help=".parquet file, the dataset"
        )
        parser.add_argument(
                "--depth"
                type = int,
                default = 4,
                help = "To know how much candle you want it to predict, for example 4 will predict the 4 next candles"
        )

        args = parser.parse_args()
        epochs = args.epochs
        layers = args.layers
        learning_rate = args.learning_rate
        depth = args.depth

        df = pd.read_parquet(args.parquet)
        for i in range(1, depths+1):
            maxname = f'target{i}'
            df[maxname] = (df['high'].shift(-i) - df['close']) / df['close']
        dataset = df.dropna()
        #splitting data and tensors
        targets = [f'target{i}' for i in range(1, depth + 1)]
        data = [col for col in dataset.columns if col not in targets]
        
        split = int(len(dataset) * 0.8)
        learn_df = dataset.iloc[:split]
        exam_df = dataset.iloc[split:]
        input_size = len(dataset.columns)

        #creation of the layer architecture + output (depth size neuron in output)
        architecture = layers + [h]
        #litteraly makes a Matrix for every line of the dataset
        input_matrix_list = [MatrixFloat(input_size, 1, e_types.NO_TYPE, [value for column, value, in learn_df.iloc[i].items()], -1, -1) for i in range(len(learn_df))]
        input_matrix_list_exam = [MatrixFloat(input_size, 1, e_types.NO_TYPE, [value for column, value, in exam_df.iloc[i].items()], -1, -1) for i in range(len(exam_df))]

        layers_weight_matrices = []
        layers_bias_matrices = []
        current_input_size = input_size
        for layer in architecture:
            bias = MatrixFloat(layer, 1, e_types.NO_TYPE, [0.0] * layer, -1, -1)
            layers_bias_matrices.append(bias)
            data = [random.gauss(0.0, math.sqrt(2.0 / current_input_size)) for _ in range(current_input_size * layer)]
            weight = MatrixFloat(layer, current_input_size, e_types.NO_TYPE, data, -1, -1)
            layers_weight_matrices.append(weight)
            current_input_size = layer
        for epoch in range(epochs):
            for X in input_matrix_list:
                

    except Exception as e:
        print (e)



if __name__ == "__main__":
    main()

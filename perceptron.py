import pandas as pd
from matrix_lib import MatrixFloat, e_types
import argparse
import math
import random
import matplotlib.pyplot as plt

pd.set_option("future.no_silent_downcasting", True)

def relu(data):
    tmp = []
    for i in range(len(data)):
        val = data[i]
        tmp.append(val if val > 0.0 else 0.01 * val)
    return tmp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 5], help="how much layers and neurons (5 5, is 2 layers of 5 neurons)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--parquet", type=str, required=True, help="Parquet data file")
    parser.add_argument("--depth", type=int, default=4, help="How much candles you want to predict in the future")
    parser.add_argument("--batch-size", type=int, default=32, help="Group data in batch to speed up learning")

    args = parser.parse_args()
    epochs = args.epochs
    layers = args.layers
    learning_rate = args.learning_rate
    depth = args.depth
    batch_size = args.batch_size

    df = pd.read_parquet(args.parquet)
    
    for i in range(1, depth+1):
        maxname = f'target{i}'
        df[maxname] = ((df['high'].shift(-i) - df['close']) / df['close']) * 100.0
        
    df = df.dropna()
    
    targets = [f'target{i}' for i in range(1, depth + 1)]
    features = [col for col in df.columns if col not in targets]
    
    for f in features:
        df[f] = (df[f] - df[f].mean()) / df[f].std()

    target_stats = {}
    for t in targets:
        mean_t = df[t].mean()
        std_t = df[t].std()
        target_stats[t] = {'mean': mean_t, 'std': std_t}
        df[t] = (df[t] - mean_t) / std_t
        
        
    split = int(len(df) * 0.8)
    learn_df = df.iloc[:split]
    exam_df = df.iloc[split:]
    input_size = len(features)

    architecture = layers + [depth]

    input_matrix_list = [MatrixFloat(input_size, 1, e_types.NO_TYPE, row.tolist(), -1, -1) for _, row in learn_df[features].iterrows()]
    target_matrix_list = [MatrixFloat(depth, 1, e_types.NO_TYPE, row.tolist(), -1, -1) for _, row in learn_df[targets].iterrows()]

    exam_matrix_list = [MatrixFloat(input_size, 1, e_types.NO_TYPE, row.tolist(), -1, -1) for _, row in exam_df[features].iterrows()]
    exam_target_matrix_list = [MatrixFloat(depth, 1, e_types.NO_TYPE, row.tolist(), -1, -1) for _, row in exam_df[targets].iterrows()]

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

    epochs_plot = []
    epochs_error = []
    exam_epochs_error = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        indexes = list(range(len(input_matrix_list)))
        random.shuffle(indexes)
        
        for batch_start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[batch_start:batch_start + batch_size]
            current_batch_size = len(batch_indexes)
            
            extracted_A = [input_matrix_list[i].getData() for i in batch_indexes]
            batch_A_data = []
            for f in range(input_size):
                for sample in extracted_A:
                    batch_A_data.append(sample[f])
            current_A = MatrixFloat(input_size, current_batch_size, e_types.NO_TYPE, batch_A_data, -1, -1)
            
            extracted_Y = [target_matrix_list[i].getData() for i in batch_indexes]
            batch_Y_data = []
            for d in range(depth):
                for sample in extracted_Y:
                    batch_Y_data.append(sample[d])
            Y = MatrixFloat(depth, current_batch_size, e_types.NO_TYPE, batch_Y_data, -1, -1)

            memory_A = [current_A]
            memory_Z = []
            
            for layer in range(len(architecture)):
                Z = layers_weight_matrices[layer].multiply_mat(current_A)
                
                bias_data = layers_bias_matrices[layer].getData()
                batched_bias_data = [val for val in bias_data for _ in range(current_batch_size)]
                batched_bias = MatrixFloat(layers_bias_matrices[layer].getRows(), current_batch_size, e_types.NO_TYPE, batched_bias_data, -1, -1)
                
                Z = Z.add_mat(batched_bias)
                memory_Z.append(Z)
                data = Z.getData()
                
                if layer == len(architecture) - 1:
                    prediction = Z
                    lossSum = 0.0
                    for k in range(depth * current_batch_size):
                        lossSum += (data[k] - batch_Y_data[k]) ** 2
                    epoch_loss += lossSum / depth
                else:
                    relued_data = relu(data)
                    current_A = MatrixFloat(Z.getRows(), current_batch_size, e_types.NO_TYPE, relued_data, -1, -1)
                    memory_A.append(current_A)

            error = prediction.sub_mat(Y).multiply_scalar(2.0 / (depth * current_batch_size))
            ones = MatrixFloat(current_batch_size, 1, e_types.NO_TYPE, [1.0] * current_batch_size, -1, -1)
            
            for backprop in range(len(architecture) -1 , -1, -1):
                transposed_A = memory_A[backprop].transpose()
                gradient = error.multiply_mat(transposed_A)
                bias_gradient = error.multiply_mat(ones)

                if backprop > 0:
                    transposed_weight = layers_weight_matrices[backprop].transpose()
                    propagated_error = transposed_weight.multiply_mat(error)
                    z_data = memory_Z[backprop - 1].getData()
                    error_matrix = MatrixFloat(len(z_data) // current_batch_size, current_batch_size, e_types.NO_TYPE, [1.0 if val > 0 else 0.01 for val in z_data], -1, -1)
                    next_error = propagated_error.hadamard(error_matrix)
                    
                layers_weight_matrices[backprop] = layers_weight_matrices[backprop].sub_mat(gradient.multiply_scalar(learning_rate))
                layers_bias_matrices[backprop] = layers_bias_matrices[backprop].sub_mat(bias_gradient.multiply_scalar(learning_rate))
                
                if backprop > 0:
                    error = next_error

        exam_loss = 0.0
        for batch_start in range(0, len(exam_matrix_list), batch_size):
            batch_indexes = range(batch_start, min(batch_start + batch_size, len(exam_matrix_list)))
            current_batch_size = len(batch_indexes)
            
            extracted_A = [exam_matrix_list[i].getData() for i in batch_indexes]
            batch_A_data = []
            for f in range(input_size):
                for sample in extracted_A:
                    batch_A_data.append(sample[f])
            current_A = MatrixFloat(input_size, current_batch_size, e_types.NO_TYPE, batch_A_data, -1, -1)
            
            extracted_Y = [exam_target_matrix_list[i].getData() for i in batch_indexes]
            batch_Y_data = []
            for d in range(depth):
                for sample in extracted_Y:
                    batch_Y_data.append(sample[d])
            Y_exam = MatrixFloat(depth, current_batch_size, e_types.NO_TYPE, batch_Y_data, -1, -1)
            
            for layer in range(len(architecture)):
                Z = layers_weight_matrices[layer].multiply_mat(current_A)
                
                bias_data = layers_bias_matrices[layer].getData()
                batched_bias_data = [val for val in bias_data for _ in range(current_batch_size)]
                batched_bias = MatrixFloat(layers_bias_matrices[layer].getRows(), current_batch_size, e_types.NO_TYPE, batched_bias_data, -1, -1)
                
                Z = Z.add_mat(batched_bias)
                data = Z.getData()
                
                if layer == len(architecture) - 1:
                    lossSum = 0.0
                    for k in range(depth * current_batch_size):
                        lossSum += (data[k] - batch_Y_data[k]) ** 2
                    exam_loss += lossSum / depth
                else:
                    relued_data = relu(data)
                    current_A = MatrixFloat(Z.getRows(), current_batch_size, e_types.NO_TYPE, relued_data, -1, -1)

        epochs_plot.append(epoch)
        mean_train_error = epoch_loss / len(input_matrix_list)
        mean_exam_error = exam_loss / len(exam_matrix_list)
        
        epochs_error.append(mean_train_error)
        exam_epochs_error.append(mean_exam_error)
        
        print(f"Epoch {epoch + 1:03d}/{epochs} | Train MSE: {mean_train_error:.6f} | Validation MSE: {mean_exam_error:.6f}")

    name_parts = [str(input_size)] + [str(l) for l in architecture]
    filename = f"neural({'-'.join(name_parts)}).txt"
    with open(filename, "w") as f:
        for i, (w, b) in enumerate(zip(layers_weight_matrices, layers_bias_matrices)):
            f.write(f"Layer {i+1} Weights ({w.getRows()} {w.getCols()}):\n")
            f.write(" ".join(map(str, w.getData())) + "\n")
            f.write(f"Layer {i+1} Biases ({b.getRows()} {b.getCols()}):\n")
            f.write(" ".join(map(str, b.getData())) + "\n")

    plt.plot(epochs_plot, epochs_error, color='red', label='Train Loss')
    plt.plot(epochs_plot, exam_epochs_error, color='blue', label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

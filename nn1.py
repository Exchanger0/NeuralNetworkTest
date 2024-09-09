# custom neural network

import numpy as np
import scipy.special as sc


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int,
                 learning_rate: float):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_func = lambda x: sc.expit(x)

    def query(self, inputs_list):
        # inputs = np.array(inputs_list).T
        hidden_inputs = np.dot(self.wih, inputs_list)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        return final_outputs

    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs),
                                     np.transpose(inputs))


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# тренировка нейронной сети

with open("mnist_dataset/mnist_train_100.csv") as training_data_file:
    training_data_list = training_data_file.readlines()

for record in training_data_list:
    all_values = record.split(",")
    inputs = (np.array(all_values[1:], dtype=np.float32) / 255 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

# тестирование нейронной сети
with open("mnist_dataset/mnist_test_10.csv") as test_data_file:
    test_data_list = test_data_file.readlines()

score_card = []
for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    print(correct_label, "истинный маркер")
    inputs = (np.array(all_values[1:], dtype=np.float32) / 255 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(label, "ответ сети")
    if label == correct_label:
        score_card.append(1)
    else:
        score_card.append(0)

print(score_card)
score_card_np = np.array(score_card)
print("эффективность =", score_card_np.sum()/score_card_np.size)

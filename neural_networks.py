import numpy as np

class NeuralNetwork():

    def __init__(self):

        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3,1)) - 1


    def sigmoid(self,x):

        return 1/(1+np.exp(-x))


    def sigmoid_derivative(self,x):

        return x*(1-x)


    def train(self, training_inputs, training_outputs, iterations):

        for i in range(iterations):

            output = self.think(training_inputs)

            error = training_outputs - output

            adjustments = np.dot(
                training_inputs.T,
                error * self.sigmoid_derivative(output)
            )

            self.synaptic_weights += adjustments


    def think(self,inputs):

        inputs = inputs.astype(float)

        return self.sigmoid(
            np.dot(inputs,self.synaptic_weights)
        )


network = NeuralNetwork()


training_inputs=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])


training_outputs=np.array([[0],[1],[1],[0]])


network.train(training_inputs,training_outputs,10000)


print(network.synaptic_weights)


print("Synaptic weights after training:")
print(network.synaptic_weights)

print('Test : ')

A = float(input("Input 1: "))
B = float(input("Input 2: "))
C = float(input("Input 3: "))


new_input = np.array([A,B,C])


print("Input data:", new_input)

prediction = network.think(new_input)

print("Output prediction:")
print(prediction)

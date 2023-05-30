class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X, y):
        self.weights = [0] * len(X[0])
        self.bias = 0
        iteration = 0

        while iteration < self.max_iterations:
            error = 0
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]
                linear_model = self.bias
                for j in range(len(xi)):
                    linear_model += self.weights[j] * xi[j]
                predicted = 1 if linear_model >= 0 else 0
                update = self.learning_rate * (yi - predicted)
                self.weights = [self.weights[j] + update * xi[j] for j in range(len(xi))]
                self.bias += update
                error += int(update != 0.0)
            if error == 0:
                break
            iteration += 1

    def predict(self, X):
        predictions = []
        for xi in X:
            linear_model = self.bias
            for j in range(len(xi)):
                linear_model += self.weights[j] * xi[j]
            prediction = 1 if linear_model >= 0 else 0
            predictions.append(prediction)
        return predictions

    def score(self, X, y):
        correct_predictions = 0
        predictions = self.predict(X)
        for i in range(len(y)):
            if predictions[i] == y[i]:
                correct_predictions += 1
        return float(correct_predictions) / float(len(y)) * 100.0


def classify_manual_input(model):
    print("Podaj wektor testowy (oddzielony przecinkami): ")
    input_str = input()
    input_arr = [float(x) for x in input_str.split(",")]
    prediction = model.predict([input_arr])[0]
    print("Klasyfikacja: ", "Iris-versicolor" if prediction == 0 else "Iris-virginica")


def evaluate(model, X, y):
    accuracy = model.score(X, y)
    print("Dokładność: %.2f%%" % accuracy)


def load_data(file):
    train_data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split(",")
            if line[-1] == "Iris-versicolor":
                line[-1] = 0
            elif line[-1] == "Iris-virginica":
                line[-1] = 1
            train_data.append([float(x) for x in line])
    return train_data

train_data = load_data('perceptron.data')
X_train = [d[:-1] for d in train_data]
y_train = [d[-1] for d in train_data]

test_data = load_data('perceptron.test.data')
X_test = [d[:-1] for d in test_data]
y_test = [d[-1] for d in test_data]

# Train perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Evaluate accuracy on test data
evaluate(perceptron, X_test, y_test)

# Classify manual input
while True:
    classify_manual_input(perceptron)

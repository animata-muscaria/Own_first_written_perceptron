import random

class Perceptron:
    # init, weights, bias und learning rate werden initialisiert

    def __init__(self,input_size, learning_rate= 0.1):
        self.weights =[random.uniform(-1,1) for _ in range(input_size)]
        self.bias =random.uniform(-1,1)
        self.learning_rate = learning_rate
        

     # weights werden multipliziert
    def predict(self,inputs):
        weighted_sum = sum(w*x for w,x in zip(self.weights,inputs)) +self.bias
        return self.activation(weighted_sum)
    # aktivierungsfunktion wird abgerufen
    def activation(self,x):
        return 1 if x>=0 else 0

    def train(self, training_data):
        while True:
            error_count = 0  # Zähler für Fehler auf 0 setzen
            
            for inputs, expected in training_data:  # trainings data durchgehen
                prediction = self.predict(inputs)   # make prediction
                error = expected - prediction       # calculate error
                
                if error != 0:  # Falls die Vorhersage falsch war, Gewichte anpassen
                    for i in range(len(self.weights)):
                        self.weights[i] = self.weights[i]+self.learning_rate*error*inputs[i] # weight correcture
                    self.bias = self.bias+self.learning_rate*error            # bias anpassung
                    error_count += 1           # Ein Fehler wurde gemacht
            
            if error_count==0: #error counter stoppt das ganze
                break
            self.learning_rate=self.learning_rate*0.99
    
   
    
p = Perceptron(2)
training_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]
p.train(training_data)
test= [[0,0],[0,1],[1,1],[1,0],[1,1]]
for pair in test:
    result=p.predict(pair)
    print(result)



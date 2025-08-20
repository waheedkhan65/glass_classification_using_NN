# Entry point: import utils, train model, evaluate
from src.utils import load_data
from src.train import train, evaluate

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("data/dataset.csv")

    model = train(X_train, y_train, epochs=100, lr=0.001)
   
    evaluate(model, X_test, y_test)
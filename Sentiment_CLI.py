import joblib

# Load the model and vectorizer
model = joblib.load('sentiment_analysis_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def preprocess_and_predict(text):
    # Vectorize the input text
    text_features = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_features)

    return prediction[0]


def main():
    print("Welcome to the Sentiment Analysis CLI!")
    print("Type 'exit' to quit the program.\n")

    while True:
        # Get user input
        user_input = input("Enter a sentence for sentiment analysis: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Process the input and get prediction
        sentiment = preprocess_and_predict(user_input)

        # Output the prediction
        print(f"Predicted Sentiment: {sentiment}\n")


if __name__ == "__main__":
    main()

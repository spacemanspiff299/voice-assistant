from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from training_data import TRAINING_DATA

class NLPEngine:
    def __init__(self):
        # Improved pipeline with better parameters
        self.model = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                max_features=1000,   # Limit features to prevent overfitting
                stop_words='english' # Remove common English stop words
            ),
            SVC(
                kernel='rbf',        # RBF kernel often works better than linear for small datasets
                C=10,               # Higher C for more complex decision boundary
                gamma='scale',      # Let sklearn choose gamma
                probability=True,
                class_weight='balanced'  # Handle class imbalance
            )
        )
        self.is_trained = False

    def train(self):
        """Trains the model on the expanded training data."""
        print("Training improved NLP model...")
        
        # Use the expanded training data
        sentences = []
        labels = []
        for intent, phrases in TRAINING_DATA.items():
            for phrase in phrases:
                sentences.append(phrase.lower())  # Normalize to lowercase
                labels.append(intent)
        
        print(f"Training on {len(sentences)} examples across {len(set(labels))} intents")
        
        self.model.fit(sentences, labels)
        self.is_trained = True
        print("Improved NLP model trained successfully.")

    def predict(self, text):
        """Predicts the intent of a given text with improved confidence handling."""
        if not self.is_trained:
            return "Model not trained yet."
        
        # Normalize input text
        text_normalized = text.lower()
        
        prediction = self.model.predict([text_normalized])[0]
        confidence_scores = self.model.predict_proba([text_normalized])[0]
        max_confidence = max(confidence_scores)
        
        # Adaptive confidence threshold based on intent
        if prediction == "None":
            CONFIDENCE_THRESHOLD = 0.4  # Lower threshold for None intent
        else:
            CONFIDENCE_THRESHOLD = 0.5  # Slightly lower threshold for commands
        
        if max_confidence >= CONFIDENCE_THRESHOLD:
            print(f"   (NLP Intent: '{prediction}', Confidence: {max_confidence:.2f})")
            return prediction
        else:
            print(f"   (NLP Unsure. Best guess: '{prediction}', Confidence: {max_confidence:.2f})")
            return None
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import numpy as np

def training():
    # File paths
    embeddings_file = "output/embeddings.pickle"
    recognizer_file = "output/recognizer.pickle"
    label_encoder_file = "output/le.pickle"

    # Load embeddings
    print("[INFO] Loading face embeddings...")
    data = pickle.loads(open(embeddings_file, "rb").read())

    # Debugging: Check unique classes
    print("[DEBUG] Unique classes:", set(data["names"]))
    print("[DEBUG] Number of unique classes:", len(set(data["names"])))

    if len(set(data["names"])) <= 1:
        raise ValueError("[ERROR] The dataset must have at least two unique classes to train the model.")

    # Convert embeddings to a NumPy array
    embeddings = np.array(data["embeddings"])

    # Encode labels
    print("[INFO] Encoding labels...")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["names"])

    # Train the recognizer
    print("[INFO] Training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(embeddings, labels)

    # Save the trained model and label encoder
    with open(recognizer_file, "wb") as f:
        f.write(pickle.dumps(recognizer))
    with open(label_encoder_file, "wb") as f:
        f.write(pickle.dumps(label_encoder))

    print("[INFO] Training complete. Model saved!")

# Run the training function
training()

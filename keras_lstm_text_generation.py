import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate training data
text = "Hello, this is a Keras-based code generation model example."
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

max_len = 40  # Maximum length of input sequences
step = 3  # Interval for extracting sequences

sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i : i + max_len])
    next_chars.append(text[i + max_len])

x = np.zeros((len(sentences), max_len, len(chars)), dtype=bool)  # Change dtype to bool
y = np.zeros((len(sentences), len(chars)), dtype=bool)  # Change dtype to bool

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Build the model
print("Building the model...")
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, len(chars))))
model.add(Dense(len(chars), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy")

# Train the model
print("Training the model...")
model.fit(x, y, epochs=50, batch_size=128)


# Text generation function
def generate_text(seed_text, length=100):
    generated_text = seed_text
    for _ in range(length):
        x_pred = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(seed_text):
            if t < max_len:  # Check if the index is within the valid range
                x_pred[0, t, char_indices[char]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        generated_text += next_char
        seed_text = seed_text[1:] + next_char

    return generated_text


# Generate text
print("\nGenerated Text:")
generated_text = generate_text(
    "Hello, this is a Keras-based code generation model example.", length=200
)
print(generated_text)

import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

# Split dataset into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data - this is important for neural networks
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple Sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test,  y_test, verbose=2)

# Save the model
model.save('./models')
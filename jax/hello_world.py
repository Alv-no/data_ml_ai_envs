"""
Application to create, from sratch, a deep neural network with JAX. 
Author: Arturo Opsetmoen Amador
Date: 30.08.2022
"""

import jax.numpy as jnp
import jax as j
from jax.scipy.special import logsumexp
import tensorflow as tf

# tf.config.set_visible_devices([], device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.compat.v1.Session(config=config)

import tensorflow_datasets as tfds


data_dir = "/tmp/tfds"
layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10


def random_layer_params(
    m: int, n: int, key: jnp.ndarray, scale: float = 1e-2
) -> jnp.ndarray:
    """
    A helper function to randomly initialize the weights and biases for a dense neural network layer.
    """
    w_key, b_key = j.random.split(key)
    return scale * j.random.normal(w_key, (n, m)), scale * j.random.normal(b_key, (n,))


def init_network_params(sizes: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """
    Initialize all layers for a fully-connected neural network with sizes "sizes".
    """
    keys = j.random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def relu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rectified linear unit in jax
    """
    return jnp.maximum(x, 0)


def predict(params: jnp.ndarray, image: jnp.ndarray) -> jnp.ndarray:
    """
    Predict the label of an image using a neural network. Per-example predictions"""
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


def one_hot(x: jnp.ndarray, k: int, dtype=jnp.float32) -> jnp.ndarray:
    """
    Create a one-hot encoding of x of size k.
    """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params: jnp.ndarray, images: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    Compute the accuracy of the neural network on a batch of examples.
    """
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params: jnp.ndarray, images: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    Compute the loss of the neural network on a batch of examples.
    """
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


@j.jit
def update(params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Update the parameters of the neural network by one step of gradient descent.
    """
    grads = j.grad(loss)(params, x, y)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


params = init_network_params(layer_sizes, j.random.PRNGKey(0))


random_flattened_image = j.random.normal(j.random.PRNGKey(1), (28 * 28,))
preds = predict(params, random_flattened_image)
print(preds.shape)

random_flattened_images = j.random.normal(j.random.PRNGKey(1), (10, 28 * 28))
try:
    preds = predict(params, random_flattened_images)
except TypeError:
    print("Invalid shapes!")


batched_predict = j.vmap(predict, in_axes=(None, 0))

batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)


mnist_data, info = tfds.load(
    name="fashion_mnist", batch_size=-1, data_dir=data_dir, with_info=True
)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data["train"], mnist_data["test"]
num_labels = info.features["label"].num_classes
h, w, c = info.features["image"].shape
num_pixels = h * w * c

# Full train set
train_images, train_labels = train_data["image"], train_data["label"]
train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
train_labels = one_hot(train_labels, num_labels)

# Full test set
test_images, test_labels = test_data["image"], test_data["label"]
test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
test_labels = one_hot(test_labels, num_labels)

print("Train:", train_images.shape, train_labels.shape)
print("Test:", test_images.shape, test_labels.shape)

import time


def get_train_batches():
    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(
        name="fashion_mnist", split="train", as_supervised=True, data_dir=data_dir
    )
    # You can build up an arbitrary tf.data input pipeline
    ds = ds.batch(batch_size).prefetch(1)
    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(ds)


for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in get_train_batches():
        x = jnp.reshape(x, (len(x), num_pixels))
        y = one_hot(y, num_labels)
        params = update(params, x, y)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

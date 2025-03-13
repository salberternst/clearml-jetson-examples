from clearml import Task

task = Task.init(project_name='jetson-nano-examples',task_name='Simple Training Tensorflow')

task.add_requirements('tensorflow', '2.7.0+nv22.1')
task.add_requirements('keras', '2.7.0')
task.add_requirements('numpy', '1.19.5')

task.set_repo(repo="https://github.com/salberternst/clearml-jetson-examples.git", branch='main')
task.set_base_docker(docker_image='nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3')
task.execute_remotely(
    queue_name='jetson-nano',
    exit_process=True
)

import tensorflow as tf
import numpy as np

x_train = np.random.random((100, 10))
y_train = np.random.randint(2, size=(100,))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=16)

model.save('simple_nn.h5')

task.close()
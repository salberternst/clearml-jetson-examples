from clearml import Task

task = Task.init(project_name='jetson-nano-examples',task_name='Simple Training Pytorch')

task.add_requirements('torch', '1.10.0')
task.add_requirements('torchaudio', '0.10.0+d2634d8')
task.add_requirements('torchvision', '0.11.0a0+fa347eb')
task.add_requirements('numpy', '1.19.5')
task.add_requirements('Pillow', '8.4.0')

task.set_repo(repo="https://github.com/salberternst/clearml-jetson-examples.git", branch='main')
task.set_base_docker(docker_image='nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3')
task.execute_remotely(
    queue_name='jetson-nano',
    exit_process=True
)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

x_train = torch.FloatTensor(np.random.random((100, 10)))
y_train = torch.LongTensor(np.random.randint(2, size=(100,)))

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

batch_size = 16
epochs = 3

for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size].float().unsqueeze(1)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
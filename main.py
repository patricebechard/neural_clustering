import torch
from torch import nn
import torch.nn.functional as F 
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import Clusterizer
from utils import generate_gaussians_mixture_dataset, generate_two_moons_dataset
from utils import use_cuda, visualize_cluster_assignment

import sys
import matplotlib.pyplot as plt

def train(model, dataset, n_epochs=20, learning_rate=1e-3, epsilon=1e-3):

	if use_cuda:
		model = model.cuda()

	criterion = nn.MSELoss()
	# criterion = nn.L1Loss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	prev_loss = 1e9
	epoch_loss = 1e8

	loss_tracker = []
	epoch = 0

	while prev_loss - epoch_loss >= epsilon:

		loss_tracker = []

		prev_loss = epoch_loss
		epoch_loss = 0

		for batch_idx, (inputs, _) in enumerate(dataset['train']):

			optimizer.zero_grad()

			inputs = Variable(inputs)
			if use_cuda:
				inputs = inputs.cuda()

			outputs = model(inputs)

			loss = criterion(outputs, inputs)
			epoch_loss += loss.data[0]
			loss.backward()
			optimizer.step()

		print("Epoch : %d ----- Loss : %.2f" % (epoch, epoch_loss/len(dataset)))

		epoch += 1

	return loss_tracker

def evaluate(model, dataset):

	n_correct = 0
	n_total = 0

	for batch_idx, (inputs, targets) in enumerate(dataset):

		inputs, targets = Variable(inputs), Variable(targets)
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()

		cluster_prob = model(inputs, eval=True)
		_, predictions = torch.max(cluster_prob, dim=-1)

		n_correct += torch.sum(torch.eq(predictions, targets)).data[0]
		n_total += len(targets)

	accuracy = (n_correct / n_total) * 100

	return accuracy

if __name__ == "__main__":

	input_size = 2
	batch_size = 32
	n_clusters = 2
	dataset_size = 10000

	# dataset, means, covs = generate_dataset(input_size=input_size, 
	# 										n_clusters=n_clusters,
	# 										dataset_size=dataset_size)

	dataset = generate_two_moons_dataset(dataset_size)

	dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True) for x in dataset.keys()}

	model = Clusterizer(input_size=input_size, n_clusters=2)

	loss = train(model, dataloader)

	visualize_cluster_assignment(model, dataset)
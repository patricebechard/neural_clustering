from utils import use_cuda

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

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

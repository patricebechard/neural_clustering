import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

def generate_dataset(input_size, n_clusters=4, span=5, dataset_size=10000):

	dataset = {'train': [], 'valid': [], 'test': []}
	cluster_assignment = []
	means = []
	covs = []

	for i in range(n_clusters):

		mean = torch.rand(input_size) * (2*span) - span
		
		# cannot generate examples from full cov matrix with torch.normal
		# using diagonal cov matrix instead
		cov = torch.randn(input_size)
		# cov = torch.randn(input_size, input_size)

		train_samples = torch.stack([torch.normal(mean, cov) for j in range(dataset_size//n_clusters)])
		valid_samples = torch.stack([torch.normal(mean, cov) for j in range(dataset_size//n_clusters)])		
		test_samples = torch.stack([torch.normal(mean, cov) for j in range(dataset_size//n_clusters)])

		means.append(mean)
		covs.append(cov)
		dataset['train'].append(train_samples)
		dataset['valid'].append(valid_samples)
		dataset['test'].append(test_samples)

	means = torch.stack(means)
	covs = torch.stack(covs)

	dataset['train'] = torch.stack(dataset['train']).view(-1, input_size)
	dataset['valid'] = torch.stack(dataset['valid']).view(-1, input_size)
	dataset['test'] = torch.stack(dataset['test']).view(-1, input_size)

	cluster_assignment = torch.floor(torch.arange(dataset_size) / (dataset_size // n_clusters)).long()

	dataset['train'] = TensorDataset(dataset['train'], cluster_assignment)
	dataset['valid'] = TensorDataset(dataset['valid'], cluster_assignment)
	dataset['test'] = TensorDataset(dataset['test'], cluster_assignment)

	return dataset, means, covs

def visualize_cluster_assignment(model, dataset):

	for dset in ['train', 'valid']:

		# plotting ground_truth
		data = dataset[dset][:][0].numpy()
		real_labels = dataset[dset][:][1].numpy()
		plt.scatter(data[:,0], data[:,1], marker='.', c=real_labels)
		plt.savefig('ground_truth_%s.png'%dset)
		plt.clf()

		pred_labels = []
		for elem, _ in dataset[dset]:
		
			elem = elem.unsqueeze(0)
			elem = Variable(elem)
			if use_cuda:
				elem = elem.cuda()

			cluster_prob = model(elem, eval=True)
			_, predictions = torch.max(cluster_prob, dim=-1)

			pred_labels.append(predictions.data[0])

		data = dataset[dset][:][0].numpy()

		# plotting predictions
		plt.scatter(data[:,0], data[:,1], marker='.', c=pred_labels)
		plt.savefig('prediction_%s.png'%dset)
		plt.clf()


if __name__ == "__main__":

	dataset, means, covs = generate_dataset(input_size=4)
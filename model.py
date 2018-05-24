from torch import nn
import torch.nn.functional as F 

class Clusterizer(nn.Module):

	def __init__(self, input_size, n_clusters, hidden_size=100):
		super(Clusterizer, self).__init__()

		self.n_clusters = n_clusters
		self.input_size = input_size

		self.encoder = Encoder(input_size=input_size, 
							   n_clusters=n_clusters,
							   hidden_size=hidden_size)
		self.decoder = Decoder(output_size=input_size, 
			                   n_clusters=n_clusters,
			                   hidden_size=hidden_size)

	def forward(self, x, eval=False):

		x = self.encoder(x)

		if eval:
			return x
		else:
			return self.decoder(x)

class Encoder(nn.Module):

	def __init__(self, input_size, n_clusters, hidden_size=100):
		super(Encoder, self).__init__()

		self.input_size = input_size
		self.n_clusters = n_clusters
		self.hidden_size = hidden_size

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, n_clusters)

	def forward(self, x):

		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x), -1)

		return x

class Decoder(nn.Module):

	def __init__(self, output_size, n_clusters, hidden_size=100):
		super(Decoder, self).__init__()

		self.n_clusters = n_clusters
		self.output_size = output_size
		self.hidden_size = hidden_size

		self.fc1 = nn.Linear(n_clusters, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):

		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return x
import torch as T 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor 
import numpy as np 
import os
import matplotlib.pyplot as plt 

class CNN(nn.Module):
	def __init__(self, lr, epochs, batch_size, name, chkpt_dir, num_classes=10):
		super(CNN, self).__init__()
		self.chkpt_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.chkpt_dir, name)
		self.epochs = epochs
		self.lr = lr
		self.batch_size = batch_size 
		self.num_classes = num_classes
		self.loss_history = []
		self.acc_history = []
		self.device = T.device("cpu")
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, 3)
		self.bn3 = nn.BatchNorm2d(32)
		self.maxpool1 = nn.MaxPool2d(2)
		self.conv4 = nn.Conv2d(32, 64, 3)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 64, 3)
		self.bn5 = nn.BatchNorm2d(64)	
		self.conv6 = nn.Conv2d(64, 64, 3)
		self.bn6 = nn.BatchNorm2d(64)		
		self.maxpool2 = nn.MaxPool2d(2)

		input_dims = self.calc_input_dims()

		self.fc1 = nn.Linear(input_dims, self.num_classes)

		self.optimizer = optim.Adam(self.parameters(), lr=self.lr) # the self.parameters() comes from nn.Module 

		self.loss = nn.CrossEntropyLoss() # since we have more than 2 classes cross entropy is best otherwise with 2 classes we might be able to use binary loss

		self.to(self.device)
		self.get_data()

	def calc_input_dims(self):
		batch_data = T.zeros((1, 1, 28, 28)) # 4-tensor of 0s and we plug it into the layer and see what comes out 
		batch_data = self.conv1(batch_data)
		# batch_data = self.bn1(batch_data) # batch norm layer does not change dimesnionality
		batch_data = self.conv2(batch_data)
		# batch_data = self.bn2(batch_data)
		batch_data = self.conv3(batch_data)
		# batch_data = self.bn3(batch_data)
		batch_data = self.maxpool1(batch_data)
		batch_data = self.conv4(batch_data)
		batch_data = self.conv5(batch_data)
		batch_data = self.conv6(batch_data)
		batch_data = self.maxpool2(batch_data)

		return int(np.prod(batch_data.size())) # this will give us the input dimesniosn 

	def forward(self, batch_data): # NOTE: we can kinda combine this with calc_input_dims() 
		batch_data = T.tensor(batch_data).to(self.device) # lower case tensor() preserves the datatype while Tensor() changes the datatype to some default datatype 
		# we do a to(self.device) in order to make sure it is not a cuda tensor 

		batch_data = self.conv1(batch_data)
		batch_data = self.bn1(batch_data) # debate about whether to do batch norm before or after relu but this works fine for this one especially since relu is a noncommutative operation with respect to things like addition
		batch_data = F.relu(batch_data)

		batch_data = self.conv2(batch_data)
		batch_data = self.bn2(batch_data)
		batch_data = F.relu(batch_data)

		batch_data = self.conv3(batch_data)
		batch_data = self.bn3(batch_data)
		batch_data = F.relu(batch_data)

		batch_data = self.maxpool1(batch_data)

		batch_data = self.conv4(batch_data)
		batch_data = self.bn4(batch_data)
		batch_data = F.relu(batch_data) 

		batch_data = self.conv5(batch_data)
		batch_data = self.bn5(batch_data)
		batch_data = F.relu(batch_data)

		batch_data = self.conv6(batch_data)
		batch_data = self.bn6(batch_data)
		batch_data = F.relu(batch_data)

		batch_data = self.maxpool2(batch_data)

		batch_data = batch_data.view(batch_data.size()[0], -1)

		classes = self.fc1(batch_data)
		# note that we are not doing another activation after this since the linear cross entropy loss performs a softmax activation on it already 

		return classes

	def get_data(self):
		mnist_train_data = MNIST("mnist", train=True, download=True, 
								transform=ToTensor())
		self.train_data_loader = T.utils.data.DataLoader(mnist_train_data, 
									batch_size=self.batch_size, shuffle=True, # always want to shuffle the in case it was not preshuffled so that we get actual learning 
									num_workers=3) # this part is just so that the computer can split up the task so make it less than 4 for a mac 

		mnist_test_data = MNIST("mnist", train=False, download=True, 
								transform=ToTensor())
		self.test_data_loader = T.utils.data.DataLoader(mnist_test_data, 
									batch_size=self.batch_size, shuffle=True, # always want to shuffle the in case it was not preshuffled so that we get actual learning 
									num_workers=3) 

	def _train(self):
		self.train() # this is important if you are using pytorch with batch norm (it only switches the neural net to a train mode where it remembers the batch norm statistics for training thus only do this with batch norm)

		for i in range(self.epochs): # iteration over the full dataset (we have 60,000 in training set, 10,000 in the test set) so we want to iterate over it many many times 
			ep_loss = 0
			ep_acc = [] # this is epoch accuracy 
			for j, (input, label) in enumerate(self.train_data_loader): # the default format is an integer and a tuple with an input and an actual label
				self.optimizer.zero_grad() # remember to always zero the gradient before your training as otherwise it will remember stuff from the last cycle 
				label = label.to(self.device)
				prediction = self.forward(input)
				loss = self.loss(prediction, label)
				prediction = F.softmax(prediction, dim=1) # the softmax is so that we get a probabilities over the classes 
				classes = T.argmax(prediction, dim=1)
				wrong = T.where(classes != label, T.tensor([1.]).to(self.device), T.tensor([0.]).to(self.device)) # this looks at when the labels are not correct and marsk those with a 1
				acc = 1 - T.sum(wrong) / self.batch_size

				ep_acc.append(acc.item()) # acc is a tensor so we look at the item in the tensor 
				self.acc_history.append(acc.item())
				ep_loss += loss.item()
				loss.backward() # this calculates the gradient and is VERY IMPORTANT 
				self.optimizer.step() # this uses the optimizer to adjust the weights ALSO VERY IMPORTANT 

				if (j % 500 == 0):
					print("Epoch ", i, "Data Point ", j, "total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))

			print("Finished Epoch ", i, "total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))
			self.loss_history.append(ep_loss)
			if (i % 5 == 4):
				self.save_checkpoint()


	def _test(self):
		# self.test() # this is important if you are using pytorch with batch norm so now we can run it in test mode

		ep_loss = 0
		ep_acc = [] # this is epoch accuracy 
		for j, (input, label) in enumerate(self.test_data_loader): # the default format is an integer and a tuple with an input and an actual label
			label = label.to(self.device)
			prediction = self.forward(input)
			loss = self.loss(prediction, label)
			prediction = F.softmax(prediction, dim=1) # the softmax is so that we get a probabilities over the classes 
			classes = T.argmax(prediction, dim=1)
			wrong = T.where(classes != label, T.tensor([1.]).to(self.device), T.tensor([0.]).to(self.device)) # this looks at when the labels are not correct and marsk those with a 1
			acc = 1 - T.sum(wrong) / self.batch_size

			ep_acc.append(acc.item()) # acc is a tensor so we look at the item in the tensor 
			ep_loss += loss.item()


		print("Total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))

	def save_checkpoint(self):
		print('... saving checkpoint ...')
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('... loading checkpoint ...')
		self.load_state_dict(T.load(self.checkpoint_file))


if __name__ == '__main__':
	network = CNN(lr=0.001, batch_size=8, epochs=25, name='CNN_V1', chkpt_dir='cnn_history')
	# network.load_checkpoint();
	network._train()
	plt.plot(network.loss_history)
	plt.show()
	plt.plot(network.acc_history)
	plt.show()
	network._test()


# class CNN2(nn.Module):
# 	def __init__(self, lr, epochs, batch_size, name, chkpt_dir, num_classes=2):
# 		super(CNN, self).__init__()
# 		self.chkpt_dir = chkpt_dir
# 		self.checkpoint_file = os.path.join(self.chkpt_dir, name)
# 		self.epochs = epochs
# 		self.lr = lr
# 		self.batch_size = batch_size 
# 		self.num_classes = num_classes
# 		self.loss_history = []
# 		self.acc_history = []
# 		self.device = T.device("cpu")
# 		self.conv1 = nn.Conv2d(1, 32, 3)
# 		self.bn1 = nn.BatchNorm2d(32)
# 		self.conv2 = nn.Conv2d(32, 32, 3)
# 		self.bn2 = nn.BatchNorm2d(32)
# 		self.conv3 = nn.Conv2d(32, 32, 3)
# 		self.bn3 = nn.BatchNorm2d(32)
# 		self.maxpool1 = nn.MaxPool2d(2)
# 		self.conv4 = nn.Conv2d(32, 64, 3)
# 		self.bn4 = nn.BatchNorm2d(64)
# 		self.conv5 = nn.Conv2d(64, 64, 3)
# 		self.bn5 = nn.BatchNorm2d(64)	
# 		self.conv6 = nn.Conv2d(64, 64, 3)
# 		self.bn6 = nn.BatchNorm2d(64)		
# 		self.maxpool2 = nn.MaxPool2d(2)

# 		input_dims = self.calc_input_dims()
# 		print(input_dims)
# 		self.fc1 = nn.Linear(input_dims, 32) 
# 		self.fc2 = nn.Linear(32, 16) 
# 		self.fc3 = nn.Linear(16, self.num_classes)
        
# 		self.optimizer = optim.Adam(self.parameters(), lr=self.lr) # the self.parameters() comes from nn.Module 

# 		self.loss = nn.BCELoss() # since we have more than 2 classes cross entropy is best otherwise with 2 classes we might be able to use binary loss

# 		self.to(self.device)
# 		self.get_data()

# 	def calc_input_dims(self):
# 		batch_data = T.zeros((1, 1, 28, 28)) # 4-tensor of 0s and we plug it into the layer and see what comes out 
# 		print(batch_data.size())
# 		batch_data = self.conv1(batch_data)
# 		print(batch_data.size())
# 		# batch_data = self.bn1(batch_data) # batch norm layer does not change dimesnionality
# 		batch_data = self.conv2(batch_data)
# 		print(batch_data.size())
# 		# batch_data = self.bn2(batch_data)
# 		batch_data = self.conv3(batch_data)
# 		print(batch_data.size())
# 		# batch_data = self.bn3(batch_data)
# 		batch_data = self.maxpool1(batch_data)
# 		print(batch_data.size())
# 		batch_data = self.conv4(batch_data)
# 		print(batch_data.size())
# 		batch_data = self.conv5(batch_data)
# 		print(batch_data.size())
# 		batch_data = self.conv6(batch_data)
# 		print(batch_data.size())
# 		batch_data = self.maxpool2(batch_data)
# 		print(batch_data.size())

# 		return int(np.prod(batch_data.size())) # this will give us the input dimesniosn 

# 	def forward(self, batch_data): # NOTE: we can kinda combine this with calc_input_dims() 
# 		batch_data = T.tensor(batch_data).to(self.device) # lower case tensor() preserves the datatype while Tensor() changes the datatype to some default datatype 
# 		# we do a to(self.device) in order to make sure it is not a cuda tensor 

# 		batch_data = self.conv1(batch_data)
# 		batch_data = self.bn1(batch_data) # debate about whether to do batch norm before or after relu but this works fine for this one especially since relu is a noncommutative operation with respect to things like addition
# 		batch_data = F.relu(batch_data)

# 		batch_data = self.conv2(batch_data)
# 		batch_data = self.bn2(batch_data)
# 		batch_data = F.relu(batch_data)

# 		batch_data = self.conv3(batch_data)
# 		batch_data = self.bn3(batch_data)
# 		batch_data = F.relu(batch_data)

# 		batch_data = self.maxpool1(batch_data)

# 		batch_data = self.conv4(batch_data)
# 		batch_data = self.bn4(batch_data)
# 		batch_data = F.relu(batch_data) 

# 		batch_data = self.conv5(batch_data)
# 		batch_data = self.bn5(batch_data)
# 		batch_data = F.relu(batch_data)

# 		batch_data = self.conv6(batch_data)
# 		batch_data = self.bn6(batch_data)
# 		batch_data = F.relu(batch_data)

# 		batch_data = self.maxpool2(batch_data)

# 		batch_data = batch_data.view(batch_data.size()[0], -1)

# 		batch_data = self.fc1(batch_data)
# 		batch_data = self.fc2(batch_data)
# 		classes = self.fc3(batch_data)

# 		# note that we are not doing another activation after this since the linear cross entropy loss performs a softmax activation on it already 

# 		return classes

# 	def get_data(self):
		# images, labels = load_images_from_folder("final_dataset")
		# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.001, random_state=42)
		# train_dataset = T.utils.data.TensorDataset(T.Tensor(X_train), T.Tensor(y_train))
		# test_dataset = T.utils.data.TensorDataset(T.Tensor(X_test), T.Tensor(y_test))
        
# 		mnist_train_data = MNIST("mnist", train=True, download=True, 
# 								transform=ToTensor())
# 		self.train_data_loader = T.utils.data.DataLoader(mnist_train_data, 
# 									batch_size=self.batch_size, shuffle=True, # always want to shuffle the in case it was not preshuffled so that we get actual learning 
# 									num_workers=3) # this part is just so that the computer can split up the task so make it less than 4 for a mac 
        
# 		mnist_test_data = MNIST("mnist", train=False, download=True, 
# 								transform=ToTensor())
# 		self.test_data_loader = T.utils.data.DataLoader(mnist_test_data, 
# 									batch_size=self.batch_size, shuffle=True, # always want to shuffle the in case it was not preshuffled so that we get actual learning 
# 									num_workers=3)
# 	def _train(self):
# 		self.train() # this is important if you are using pytorch with batch norm (it only switches the neural net to a train mode where it remembers the batch norm statistics for training thus only do this with batch norm)

# 		for i in range(self.epochs): # iteration over the full dataset (we have 60,000 in training set, 10,000 in the test set) so we want to iterate over it many many times 
# 			ep_loss = 0
# 			ep_acc = [] # this is epoch accuracy 
# 			for j, (input, label) in enumerate(self.train_data_loader): # the default format is an integer and a tuple with an input and an actual label
# 				self.optimizer.zero_grad() # remember to always zero the gradient before your training as otherwise it will remember stuff from the last cycle 
# 				label = label.to(self.device)
# 				print(input.size())
# 				prediction = self.forward(input)
# 				loss = self.loss(prediction, label)
# 				prediction = F.softmax(prediction, dim=1) # the softmax is so that we get a probabilities over the classes 
# 				classes = T.argmax(prediction, dim=1)
# 				wrong = T.where(classes != label, T.tensor([1.]).to(self.device), T.tensor([0.]).to(self.device)) # this looks at when the labels are not correct and marsk those with a 1
# 				acc = 1 - T.sum(wrong) / self.batch_size

# 				ep_acc.append(acc.item()) # acc is a tensor so we look at the item in the tensor 
# 				self.acc_history.append(acc.item())
# 				ep_loss += loss.item()
# 				loss.backward() # this calculates the gradient and is VERY IMPORTANT 
# 				self.optimizer.step() # this uses the optimizer to adjust the weights ALSO VERY IMPORTANT 

# 				if (j % 500 == 0):
# 					print("Epoch ", i, "Data Point ", j, "total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))

# 			print("Finished Epoch ", i, "total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))
# 			self.loss_history.append(ep_loss)
# 			if (i % 5 == 4):
# 				self.save_checkpoint()


# 	def _test(self):
# 		# self.test() # this is important if you are using pytorch with batch norm so now we can run it in test mode

# 		ep_loss = 0
# 		ep_acc = [] # this is epoch accuracy 
# 		for j, (input, label) in enumerate(self.test_data_loader): # the default format is an integer and a tuple with an input and an actual label
# 			label = label.to(self.device)
# 			prediction = self.forward(input)
# 			loss = self.loss(prediction, label)
# 			prediction = F.softmax(prediction, dim=1) # the softmax is so that we get a probabilities over the classes 
# 			classes = T.argmax(prediction, dim=1)
# 			wrong = T.where(classes != label, T.tensor([1.]).to(self.device), T.tensor([0.]).to(self.device)) # this looks at when the labels are not correct and marsk those with a 1
# 			acc = 1 - T.sum(wrong) / self.batch_size

# 			ep_acc.append(acc.item()) # acc is a tensor so we look at the item in the tensor 
# 			ep_loss += loss.item()


# 		print("Total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))

# 	def save_checkpoint(self):
# 		print('... saving checkpoint ...')
# 		T.save(self.state_dict(), self.checkpoint_file)

# 	def load_checkpoint(self):
# 		print('... loading checkpoint ...')
# 		self.load_state_dict(T.load(self.checkpoint_file))






























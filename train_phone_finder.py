import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

import util
import dataLoader

def main():
	# training params
	lr = 0.0001
	batch_size = 4
	num_epochs = 1
	root = sys.argv[1]

	# test if gpu should be used
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# data augmentation transformations
	train_transformations = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32,padding=4),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	# train dataset loader
	train_data_loader = dataLoader.TrainingSetLoader(root, train_transformations)
	train_loader = torch.utils.data.DataLoader(dataset=train_data_loader,
													batch_size=batch_size,
													shuffle=True)
	# can add test loader here


	from simpleCNN import Net 
	model = Net().to(device)
	# can also use SGD and add momentum
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.BCELoss()

	model.train()
	for epoch in range(num_epochs):
		# Randomly shuffle data every epoch
		train_accu = []
		for batch_idx, (data, target) in enumerate(train_loader, 0):
			data = data.type(torch.FloatTensor)
			target = target.type(torch.FloatTensor)
			data, target = Variable(data), Variable(target)
			data, target = data.to(device), target.to(device)

			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			prediction = (output >= 0.5).type(torch.FloatTensor).to(device) # first column has actual prob.
			accuracy =  (prediction.eq(target.view(prediction.shape)).sum().item()/batch_size) * 100
			train_accu.append(accuracy)

		accuracy_epoch = np.mean(train_accu)
		print(str(epoch+1), accuracy_epoch)
	
	# only save the params
	# torch.save(model.state_dict(), 'trained.pth')
	# save the whole model
	torch.save(model, 'trained.pth')
	print('======= weights saved as trained.pth =======')

if __name__ == '__main__':
	main()
from model import net,dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg_ = net.VGG()
vgg_ = vgg_.to(device)
param = list(vgg_.parameters())
print(len(param))
for i in param:
    print(i.shape)

classes =  ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(vgg_.parameters(),lr=0.00001)

# get some random training images
dataiter = iter(dataloader.trainloader)
images, labels = dataiter.next()

# show images
utils.imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader.trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        #print(inputs.shape)
        #print(inputs.shape)
        # forward + backward + optimize
        outputs,f = vgg_(inputs)
        #print(outputs.shape)
        #print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if(loss.item() > 1000):
            print(loss.item())
            for param in vgg_.parameters():
                print(param.data)
        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('Finished Training')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in dataloader.testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs,_ = vgg_(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


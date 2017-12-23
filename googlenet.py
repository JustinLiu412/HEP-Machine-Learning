from __future__ import print_function
import pdb
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from data_loader import LCDDataset

num_epochs = 10
batch_size = 200 # 100
learning_rate = 0.00001
test_batch_size = 10
epsilon = 1e-05

train_loader = LCDDataset(['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_0.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_0.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_1.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_1.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_2.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_2.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_3.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_3.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_4.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_4.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_5.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_5.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_6.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_6.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_7.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_7.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_8.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_8.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_9.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_9.h5', 

                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_10.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_10.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_11.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_11.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_12.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_12.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_13.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_13.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_14.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_14.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_15.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_15.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_16.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_16.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_17.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_17.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_18.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_18.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_19.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_19.h5'
                           ], 
                          batch_size, 
                          {'Pi0': 0, 'Gamma': 1}, 
                          normalize=False, phase='Train')

test_dataset = LCDDataset(['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_0.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_0.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_1.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_1.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_2.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_2.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_3.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_3.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_4.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_4.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_5.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_5.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_6.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_6.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_7.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_7.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_8.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_8.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_9.h5', 
                           '/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_9.h5'
                           ], 
                          test_batch_size, {'Pi0': 0, 'Gamma': 1}, normalize=False, phase='Test')

# train_loader = LCDDataset(['/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_0.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_0.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_1.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_1.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_2.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_2.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_3.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_3.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_4.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_4.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_5.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_5.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_6.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_6.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_7.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_7.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_8.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_8.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_9.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_9.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_10.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_10.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_11.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_11.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_12.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_12.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_13.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_13.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_14.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_14.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_15.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_15.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_16.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_16.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_17.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_17.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_18.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_18.h5',
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTrain/ChPi_19.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTrain/Ele_19.h5'
#                            ], 
#                           batch_size, 
#                           {'ChPi': 0, 'Ele': 1}, 
#                           normalize=True, phase='Train')

# test_dataset = LCDDataset(['/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_0.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_0.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_1.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_1.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_2.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_2.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_3.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_3.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_4.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_4.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_5.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_5.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_6.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_6.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_7.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_7.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_8.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_8.h5', 
#                            '/data/LCD/V2/SimpleEleChPi/ChPiTest/ChPi_9.h5', 
#                            '/data/LCD/V2/SimpleGammaPi0/EleTest/Ele_9.h5'
#                            ], 
#                           test_batch_size, {'ChPi': 0, 'Ele': 1}, normalize=True, phase='Test')

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes, eps=epsilon),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192, eps=epsilon),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ------------------------------------------------------------------

records_train = {'epoch': [], 'accuracy': [], 'loss': []}
# records_test = {'predicted_ChPi': 0, 'true_ChPi': 0, 'predicted_Ele': 0, 'true_Ele': 0}
records_test = {'predicted_Pi0': 0, 'true_Pi0': 0, 'predicted_Gamma': 0, 'true_Gamma': 0}

def train_recorder(epoch, accuracy, loss): 
    """
    Prepare list of training data for pickle
    """
    global records_train

    records_train['epoch'].append(epoch)
    records_train['accuracy'].append(accuracy)
    records_train['loss'].append(loss)

def test_recorder(predicted_class0, true_class0, predicted_class1, true_class1): 
    """
    Prepare list of test data for pickle
    """
    global records_test
    
    # records_test['predicted_ChPi'] += predicted_Pi0
    # records_test['true_ChPi'] += true_Pi0
    # records_test['predicted_Ele'] += predicted_Gamma
    # records_test['true_Ele'] += true_Gamma
    records_test['predicted_Pi0'] += predicted_class0
    records_test['true_Pi0'] += true_class0
    records_test['predicted_Gamma'] += predicted_class1
    records_test['true_Gamma'] += true_class1

net = torch.nn.DataParallel(GoogLeNet(), device_ids=[0,1,2,3,4,5,6,7,8,9]).cuda()
# net.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


start_time = time.time()

# Train the Model
for epoch in range(num_epochs):
    # for i, (images, labels) in enumerate(train_loader):
    temp_time = time.time()
    i = 0
    correct = 0
    total = 0
    sum_loss = 0
    for _ in range(train_loader.__len__() / batch_size):
    # for _ in range(1): 
        images, labels = train_loader.getbatch()
        # images = torch.from_numpy(train_loader.tmp_data_provider.first().generate().next()[0]).cuda()
        images = images.float()
        images = Variable(images).cuda()
        # images = torch.nn.DataParallel
        # labels = train_loader.tmp_data_provider.first().generate().next()[2][0]
        saved_for_validation = labels
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)

        total += saved_for_validation.size(0)
        correct += (predicted.cpu() == saved_for_validation).sum()

        i += 1

        if (i) % 20 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, (i)*batch_size, train_loader.__len__(), loss.data[0]))
        
        sum_loss += loss.data[0]

    train_recorder(epoch, float(correct)/float(total), sum_loss / float(i))
    print('Epoch time: %s seconds' %(time.time() - temp_time))
    print('Accuracy of the model on the Epoch [%d/%d]: %d %%' % (epoch+1, num_epochs, 100 * float(correct) / float(total)))
print("--- Training time: %s seconds ---" % (time.time() - start_time))
# Save training model
torch.save(net, 'googlenet.pt')

# Save training data in pickle file
with open('googlenet_train.pickle', 'wb') as handle: 
    pickle.dump(records_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Test the Model
net.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0

for i in range(test_dataset.__len__() / test_batch_size): 
# for _ in range(1): 
# for images, labels in test_loader:
    images, labels = test_dataset.getbatch()
    images = images.float()
    images = Variable(images).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print('Tested: %d/%d' %(i*test_batch_size, test_dataset.__len__()), end='\r')
    try: 
        test_recorder(predicted.size(0)-predicted.nonzero().size(0), labels.size(0)-labels.nonzero().size(0), 
            predicted.nonzero().size(0), labels.nonzero().size(0))
    except: 
        if (not predicted.nonzero().size()) and (not labels.nonzero().size()): # predicted lables are all 0
            test_recorder(predicted.size(0), labels.size(0), 
                0, 0)
        elif not labels.nonzero().size(): 
            test_recorder(predicted.size(0)-predicted.nonzero().size(0), labels.size(0), 
                predicted.nonzero().size(0), 0)
        elif not predicted.nonzero().size(): 
            test_recorder(predicted.size(0), labels.size(0)-labels.nonzero().size(0), 
                0, labels.nonzero().size(0))

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * float(correct) / float(total)))
print("--- %s seconds ---" % (time.time() - start_time))
# Save the Trained Model
torch.save(net.state_dict(), 'googlenet_eval.pkl')

# Save test data in pickle file
with open('googlenet_test.pickle', 'wb') as handle: 
    pickle.dump(records_test, handle, protocol=pickle.HIGHEST_PROTOCOL)





# x = torch.randn(1,3,32,32)
# y = net(Variable(x))
# print(y.size())
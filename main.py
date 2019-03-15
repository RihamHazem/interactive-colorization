import numpy as np
import torch
from math import log10
from network import Unet
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn as nn
from torch.autograd import Variable
from options import parser
from data import load_data, get_samplers
from utils import preprocessing, lab2rgb

opt = parser.parse_args()

print(opt)

torch.manual_seed(opt.seed)

# if opt.cuda and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if opt.cuda else "cpu")

# loader = torch.utils.data.TensorDataset(ic)
data = load_data()
dataset_size = len(data)
print(dataset_size)

train_sampler, valid_sampler = get_samplers(dataset_size)

print('===> Loading datasets')
training_data_loader = DataLoader(dataset=data, num_workers=opt.threads, batch_size=opt.batchSize,
                                  sampler=train_sampler)
testing_data_loader = DataLoader(dataset=data, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                 sampler=valid_sampler)

print('===> Building model')
model = Unet().to(device)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


def train(epoch):
    epoch_loss = 0
    for iteration, data in enumerate(training_data_loader, 1):
        inputs, ground_truth = preprocessing(data.to(device))

        optimizer.zero_grad()
        output = model(inputs)

        # new_out = np.zeros((inputs.shape[0], 3, inputs.shape[2], inputs.shape[3]))
        # new_out[:, 0] = inputs[:, 0]
        # new_out[:, 1] = output.data.numpy()[:, 0]
        # new_out[:, 2] = output.data.numpy()[:, 1]
        #
        # output = lab2rgb(new_out)

        loss = criterion(output.float(), ground_truth.float())
        loss = Variable(loss, requires_grad=True)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    criterion2 = nn.MSELoss()
    with torch.no_grad():
        for data in testing_data_loader:
            inputs, ground_truth = preprocessing(data.to(device))

            optimizer.zero_grad()
            output = model(inputs)

            new_out = np.zeros((inputs.shape[0], 3, inputs.shape[2], inputs.shape[3]))
            new_out[:, 0] = inputs[:, 0]
            new_out[:, 1] = output.data.numpy()[:, 0]
            new_out[:, 2] = output.data.numpy()[:, 1]

            output = lab2rgb(new_out)

            mse = criterion2(output, ground_truth)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    scheduler.step()


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)

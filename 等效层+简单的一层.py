import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE=80
batch_size=64
N_pixels=80
pixel_size = 400e-9
distance = 30e-6
wl = 528e-9

def filter_digitsXX(dataset, digits=(0, 1)):
    filtered_data = []
    for item in dataset:
        if item[1] in digits:
            filtered_data.append(item)
    return filtered_data
def filter_digitsXY(dataset, digits=(2, 3)):
    filtered_data = []
    for item in dataset:
        if item[1] in digits:
            listitem = list(item)
            if item[1]==3:
                item = list(item)
                item[1] = 1
                item = tuple(item)
            else:
                item = list(item)
                item[1] = 0
                item = tuple(item)
            filtered_data.append(item)
    return filtered_data
def filter_digitsYX(dataset, digits=(4, 5)):
    filtered_data = []
    for item in dataset:
        if item[1] in digits:
            if item[1] in digits:
                if item[1] == 5:
                    item = list(item)
                    item[1] = 1
                    item = tuple(item)
                else:
                    item = list(item)
                    item[1] = 0
                    item = tuple(item)
                filtered_data.append(item)
    return filtered_data
def filter_digitsYY(dataset, digits=(6, 7)):
    filtered_data = []
    for item in dataset:
        if item[1] in digits:
            if item[1] in digits:
                if item[1] == 7:
                    item = list(item)
                    item[1] = 1
                    item = tuple(item)
                else:
                    item = list(item)
                    item[1] = 0
                    item = tuple(item)
                filtered_data.append(item)
    return filtered_data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((IMG_SIZE,IMG_SIZE))])
                                # transforms.Normalize((0.1307,), (0.3081,))

train_dataset = torchvision.datasets.MNIST("D:\pythonProject\ONN网络\\4个通道复用/data", train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.MNIST("D:\pythonProject\ONN网络\\4个通道复用/data", train=False, transform=transform, download=False)
train_filteredXX = filter_digitsXX(train_dataset)
test_filteredXX  = filter_digitsXX(test_dataset)
train_filteredXY = filter_digitsXY(train_dataset)
test_filteredXY  = filter_digitsXY(test_dataset)
train_filteredYX = filter_digitsYX(train_dataset)
test_filteredYX  = filter_digitsYX(test_dataset)
train_filteredYY = filter_digitsYY(train_dataset)
test_filteredYY  = filter_digitsYY(test_dataset)
train_dataloaderXX = torch.utils.data.DataLoader(dataset=train_filteredXX, batch_size=batch_size, shuffle=True)
test_dataloaderXX = torch.utils.data.DataLoader(dataset=test_filteredXX, batch_size=batch_size, shuffle=False)
train_dataloaderXY = torch.utils.data.DataLoader(dataset=train_filteredXY, batch_size=batch_size, shuffle=True)
test_dataloaderXY = torch.utils.data.DataLoader(dataset=test_filteredXY, batch_size=batch_size, shuffle=False)
train_dataloaderYX = torch.utils.data.DataLoader(dataset=train_filteredYX, batch_size=batch_size, shuffle=True)
test_dataloaderYX = torch.utils.data.DataLoader(dataset=test_filteredYX, batch_size=batch_size, shuffle=False)
train_dataloaderYY = torch.utils.data.DataLoader(dataset=train_filteredYY, batch_size=batch_size, shuffle=True)
test_dataloaderYY = torch.utils.data.DataLoader(dataset=test_filteredYY, batch_size=batch_size, shuffle=False)
train_dataloader=zip(train_dataloaderXX,train_dataloaderXY,train_dataloaderYX,train_dataloaderYY)
test_dataloader=zip(test_dataloaderXX,test_dataloaderXY,test_dataloaderYX,test_dataloaderYY)
def phase(J11_1,J12_1,J21_1,J22_1,J11_2,J12_2,J21_2,J22_2):
        for m in range(80):
            for n in range(80):
                x_range = torch.linspace(0, 79, 80) * pixel_size
                y_range = torch.linspace(0, 79, 80) * pixel_size
                x_range.to(device)
                y_range.to(device)
                x, y = torch.meshgrid(x_range, y_range)
                r = torch.sqrt((x - m * pixel_size)**2 + (y - n * pixel_size)**2 + distance**2).to(device)
                f = 1 / (2 * torch.pi) * torch.exp(1j * 2 * torch.pi / wl * r) / r * distance / r*(
                    1 / r - 1j * 2 * torch.pi / wl).to(device)
                g11_1=J11_1*f
                g11_2=J21_1*f
                g12_1=J12_1*f
                g12_2=J22_1*f
                g21_1=J11_1*f
                g21_2=J21_1*f
                g22_1=J12_1*f
                g22_2=J22_1*f
                J11,J12,J21,J22=J11_1,J12_1,J21_1,J22_1
                J11[m,n]=J11_2[m,n]*torch.sum(g11_1)*1/80*1/80+J12_2[m,n]+torch.sum(g11_2)*1/80*1/80
                J12[m, n] = J11_2[m, n] * torch.sum(g12_1) * 1 / 80 * 1 / 80 + J12_2[m, n] + torch.sum(g12_2) * 1 / 80 * 1 / 80
                J21[m, n] = J21_2[m, n] * torch.sum(g21_1) * 1 / 80 * 1 / 80 + J22_2[m, n] + torch.sum(g21_2) * 1 / 80 * 1 / 80
                J21[m, n] = J21_2[m, n] * torch.sum(g22_1) * 1 / 80 * 1 / 80 + J22_2[m, n] + torch.sum(g22_2) * 1 / 80 * 1 / 80
                phase11 = torch.angle(J11)
                phase12 = torch.angle(J12)
                phase21 = torch.angle(J21)
                phase22 = torch.angle(J22)
                return [phase11.to(device),phase12.to(device),phase21.to(device),phase22.to(device)]

def generate_det(length,width,start_x,start_y,cycle,N_det):
    p = []
    for i in range(len(N_det)):
        for j in range(N_det[i]):
         left = start_x+j*(cycle+length)
         right = left+length
         up = start_y[i]
         down = start_y[i]+width
         p.append((up,down,left,right))
    return list(p)
monitor8 = generate_det(10,10,10,[4,20,45,65],20,[2,2,2,2])
def monitor_region(Int,index):
    detectors_list = []
    full_Int = Int.sum(dim=(1,2))
    #print(type(monitor8),type(index))
    for det_x0, det_x1, det_y0, det_y1 in monitor8[2*index:2*index+2]: # 计算各个探测器区间内的光强占比
        detectors_list.append((Int[:, det_x0 : det_x1, det_y0 : det_y1].sum(dim=(1, 2))/full_Int).unsqueeze(-1))
    return torch.cat(detectors_list, dim = 1)
labels_image_tensorsXX=torch.zeros((2,N_pixels,N_pixels), device = device, dtype = torch.double)
labels_image_tensorsXY=torch.zeros((2,N_pixels,N_pixels), device = device, dtype = torch.double)
labels_image_tensorsYX=torch.zeros((2,N_pixels,N_pixels), device = device, dtype = torch.double)
labels_image_tensorsYY=torch.zeros((2,N_pixels,N_pixels), device = device, dtype = torch.double)
for ind, pos in enumerate(monitor8):
    pos_l, pos_r, pos_u, pos_d = pos
    if int in (0,1):
        labels_image_tensorsXX[ind, pos_l:pos_r, pos_u:pos_d] = 1
        labels_image_tensorsXX[ind] = labels_image_tensorsXX[ind]/labels_image_tensorsXX[ind].sum()
    if int in (2, 3):
        labels_image_tensorsXY[ind, pos_l:pos_r, pos_u:pos_d] = 1
        labels_image_tensorsXY[ind] = labels_image_tensorsXY[ind] / labels_image_tensorsXY[ind].sum()
    if int in (4, 5):
        labels_image_tensorsYX[ind, pos_l:pos_r, pos_u:pos_d] = 1
        labels_image_tensorsYX[ind] = labels_image_tensorsYX[ind] / labels_image_tensorsYX[ind].sum()
    if int in (6, 7):
        labels_image_tensorsYY[ind, pos_l:pos_r, pos_u:pos_d] = 1
        labels_image_tensorsYY[ind] = labels_image_tensorsYY[ind] / labels_image_tensorsYY[ind].sum()
class Equivalent_Layer(torch.nn.Module):
    def __init__(self, λ=532e-9, N_pixels=80,pixel_size=400e-9, distance=torch.tensor([0.002])):
        super(Equivalent_Layer, self).__init__()  # 初始化父类
        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fxx, fyy = np.meshgrid(fx, fy)  # 拉网格，每个网格坐标点为空间频率各分量。

        argument = (2 * np.pi) ** 2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)
        # 计算传播场或倏逝场的模式kz，传播场kz为实数，倏逝场kz为复数
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j * tmp)).to(device)

    def forward(self, E):
        fft_c = torch.fft.fft2(E)  # 对电场E进行二维傅里叶变换
        c = torch.fft.fftshift(fft_c)
        phase = torch.exp(1j * self.kz * self.distance).to(device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase))  # 卷积后逆变换得到响应的角谱
        return angular_spectrum
class Diffractive_Layer(torch.nn.Module):
    def __init__(self, λ=532e-9, N_pixels=80,pixel_size=400e-9, distance=torch.tensor([0.002])):
        super(Diffractive_Layer, self).__init__()  # 初始化父类
        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fxx, fyy = np.meshgrid(fx, fy)  # 拉网格，每个网格坐标点为空间频率各分量。

        argument = (2 * np.pi) ** 2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)
        # 计算传播场或倏逝场的模式kz，传播场kz为实数，倏逝场kz为复数
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j * tmp)).to(device)

    def forward(self, E):
        fft_c = torch.fft.fft2(E)  # 对电场E进行二维傅里叶变换
        c = torch.fft.fftshift(fft_c)
        phase = torch.exp(1j * self.kz * self.distance).to(device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase))  # 卷积后逆变换得到响应的角谱
        return angular_spectrum
class DNN(torch.nn.Module):
    """""""""""""""""""""
    phase & amplitude modulation
    """""""""""""""""""""

    def __init__(self,  Amp=[], num_layers=1, wl=532e-9, N_pixels=80, pixel_size=400e-9,
                 distance=[]):

        super(DNN, self).__init__()
        self.phi1 = [torch.nn.Parameter((torch.from_numpy((np.random.random(size=(N_pixels, N_pixels))*2*np.pi).
                                                          astype('float32'))).to(device)) for _ in range(num_layers)]
        self.phi2 = [torch.nn.Parameter((torch.from_numpy((np.random.random(size=(N_pixels, N_pixels)) * 2 * np.pi).
                                                         astype('float32'))).to(device)) for _ in range(num_layers)]
        self.theta = [torch.nn.Parameter((torch.from_numpy((np.random.random(size=(N_pixels, N_pixels))* 2 * np.pi).
                                                         astype('float32'))).to(device)) for _ in range(num_layers-1)]
        for i in range(num_layers):
            self.register_parameter("phi1" + "_" + str(i), self.phi1[i])
            self.register_parameter("phi2" + "_" + str(i), self.phi2[i])
        for i in range(num_layers-1):
            self.register_parameter("theta" + "_" + str(i), self.phi1[i])
        # print(self.phi1[0],type(self.phi1[0]))
        self.J11_1 = torch.cos(self.theta[0]) ** 2 * torch.exp(1j * self.phi1[0]) + torch.sin(self.theta[0]) ** 2 * torch.exp(1j * self.phi2[0])
        self.J12_1 = torch.sin(self.theta[0]) * torch.cos(self.theta[0]) * (torch.exp(1j * self.phi2[0]) - torch.exp(1j * self.phi1[0]))
        self.J21_1 = torch.sin(self.theta[0]) * torch.cos(self.theta[0]) * (torch.exp(1j * self.phi2[0]) - torch.exp(1j * self.phi1[0]))
        self.J22_1 = torch.cos(self.theta[0]) ** 2 * torch.exp(1j * self.phi2[0]) + torch.sin(self.theta[0]) ** 2 * torch.exp(1j * self.phi1[0])
        self.J11_2 = torch.cos(self.theta[1]) ** 2 * torch.exp(1j * self.phi1[1]) + torch.sin(
            self.theta[1]) ** 2 * torch.exp(1j * self.phi2[1])
        self.J12_2 = torch.sin(self.theta[1]) * torch.cos(self.theta[1]) * (
                    torch.exp(1j * self.phi2[1]) - torch.exp(1j * self.phi1[1]))
        self.J21_2 = torch.sin(self.theta[1]) * torch.cos(self.theta[1]) * (
                    torch.exp(1j * self.phi2[1]) - torch.exp(1j * self.phi1[1]))
        self.J22_2 = torch.cos(self.theta[1]) ** 2 * torch.exp(1j * self.phi2[1]) + torch.sin(
            self.theta[1]) ** 2 * torch.exp(1j * self.phi1[2])
        self.J11_3 = torch.exp(1j*self.phi1[2])
        self.J22_3 = torch.exp(1j * self.phi2[2])
        self.phase=phase(self.J11_1,self.J12_1,self.J21_1,self.J22_1,self.J11_2,self.J12_2,self.J21_2,self.J22_2)
        # 定义中间的衍射层
        # self.Equivalent_Layer = torch.nn.ModuleList([Equivalent_Layer(wl, N_pixels, pixel_size, distance)
        #                                                for i in range(1)])
        self.Equivalent_Layer = Equivalent_Layer(wl, N_pixels, pixel_size, distance)
        self.last_diffractive_layer = Diffractive_Layer(wl, N_pixels, pixel_size, distance)
        self.sofmax = torch.nn.Softmax(dim=-1)

    # 计算多层衍射前向传播
    def forward(self, image):
        index=0
        image1=image
        label1=torch.randn(2,2).to(device)
        for E in image:
            temp = self.Equivalent_Layer(E)
            constr_phase1 = 2 * torch.pi * self.phase[index]
            exp_j_phase1 = torch.exp(1j * constr_phase1) # torch.cos(constr_phase)+1j*torch.sin(constr_phase)
            # E = temp * exp_j_phase1
            E=E*torch.exp(1j * self.phase[index])
            E = self.last_diffractive_layer(E)
            if index==2 or index==0:
                constr_phase2 = self.J11_3
            else:
                constr_phase2 = self.J22_3
            exp_j_phase2 = constr_phase2
            E = E * exp_j_phase2
            E = self.last_diffractive_layer(E)
            Int = torch.abs(E) ** 2
            output = monitor_region(Int,index)
            image[index]=Int
            label1=torch.cat((label1,output),0)
            index+=1
        return label1[2:,:], image1[0],image1[1],image1[2],image1[3]

def train1(model, loss_function, optimizer, trainloaderXX,trainloaderXY,trainloaderYX,trainloaderYY, testloaderXX,testloaderXY,testloaderYX,testloaderYY, epochs=10, device=device):
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    best_acc = 0
    for epoch in range(epochs):
        ep_loss = 0
        # 每个epoch开始时启动Batch_Normalization和Dropout。BN层能够用到每一批数据的均值和方差，Dropout随机取一部分网络连接来训练更新参数。
        model.train()
        correct1 = 0
        total1 = 0
        # 加载进度条
        trainloader=zip(trainloaderXX,trainloaderXY,trainloaderYX,trainloaderYY)
        testloader=zip(testloaderXX,testloaderXY,testloaderYX,testloaderYY)
        for loader in tqdm(trainloader):

            imagesXX,imagesXY,imagesYX,imagesYY=loader[0][0].squeeze().to(device),loader[1][0].squeeze().to(device),loader[2][0].squeeze().to(device),loader[3][0].squeeze().to(device)
            labelsXX,labelsXY,labelsYX,labelsYY=loader[0][1].to(device),loader[1][1].to(device),loader[2][1].to(device),loader[3][1].to(device)
            labels=torch.cat((labelsXX,labelsXY,labelsYX,labelsYY),0)
            # det_labels = F.one_hot(labels, num_classes=10).to(dtype=torch.float64)
            det_labelsXX = F.one_hot(labelsXX, num_classes=2).to(dtype=torch.float64)
            det_labelsXY = F.one_hot(labelsXY, num_classes=2).to(dtype=torch.float64)
            det_labelsYX = F.one_hot(labelsYX, num_classes=2).to(dtype=torch.float64)
            det_labelsYY = F.one_hot(labelsYY, num_classes=2).to(dtype=torch.float64)
            #det_labels = labels_image_tensors[labels]
            optimizer.zero_grad()  # 梯度清零
            out_label, out_imgXX,out_imgXY,out_imgYX,out_imgYY = model([imagesXX,imagesXY,imagesYX,imagesYY])
            # _, predictedXX = torch.max(out_labelXX.data, 1)
            # _, predictedXY = torch.max(out_labelXY.data, 1)
            # _, predictedYX = torch.max(out_labelYX.data, 1)
            # _, predictedYY = torch.max(out_labelYY.data, 1)
            _, predicted = torch.max(out_label.data, 1)
            correct1 = correct1+(predicted == labels).sum().item()
            total1=total1+labelsXX.size(0)+labelsXY.size(0)+labelsYX.size(0)+labelsYY.size(0)  # 得到一个batch的标签总数
            full_int_imgXX = out_imgXX.sum(axis=(1, 2))
            # print((out_img / full_int_img[:, None, None]).shape,det_labels.shape)
            #loss = loss_function(out_img / full_int_img[:, None, None], det_labels)  # 光强分布归一化后送入损失函数（与完美探测结果进行比较）
            # loss=0
            # for index in range(out_imgXX.shape[0]):
            #     img = out_imgXX[index] * labels_image_tensorsXX[labelsXX[index]]
            #     loss = loss + loss_function(img, out_imgXX[index])
            #     #print(loss)
            # for index in range(out_imgXY.shape[0]):
            #     img = out_imgXY[index] * labels_image_tensorsXY[labelsXY[index]]
            #     loss = loss + loss_function(img, out_imgXY[index])
            # for index in range(out_imgYX.shape[0]):
            #     img = out_imgYX[index] * labels_image_tensorsYX[labelsYX[index]]
            #     loss = loss + loss_function(img, out_imgYX[index])
            # for index in range(out_imgYY.shape[0]):
            #     img = out_imgYY[index] * labels_image_tensorsYY[labelsYY[index]]
            #     loss = loss + loss_function(img, out_imgYY[index])
            #loss = loss_function(out_labelXX, det_labelsXX)+loss_function(out_labelXY, det_labelsXY)+loss_function(out_labelYX, det_labelsYX)+loss_function(out_labelYY, det_labelsYY)#(64,10)
            #out_label=torch.cat((out_labelXX,out_labelXY,out_labelYX,out_labelYY),0)
            det_labels=torch.cat((det_labelsXX,det_labelsXY,det_labelsYX,det_labelsYY),0)
            loss = loss_function(out_label, det_labels)
            loss.backward(retain_graph=True)  # 反向传播
            optimizer.step()  # 参数更新
            ep_loss += loss.item()  # 更新本次epoch的损失
        # train_loss_hist.append(ep_loss / len(trainloader))  # 计算平均损失
        print(total1)
        train_acc_hist.append(correct1 / total1)  # 计算准确率
        ep_loss = 0
        # 不启用Batch Normalization和Dropout。测试过程中要保证BN层的均值和方差不变，且利用到了所有网络连接，即不进行随机舍弃神经元。
        model.eval()
        correct2 = 0
        total2 = 0
        with torch.no_grad():  # 停止梯度更新
            for loader in tqdm(testloader):
                imagesXX, imagesXY, imagesYX, imagesYY = loader[0][0].squeeze().to(device), loader[1][0].squeeze().to(
                    device), loader[2][0].squeeze().to(device), loader[3][0].squeeze().to(device)
                labelsXX, labelsXY, labelsYX, labelsYY = loader[0][1].to(device), loader[1][1].to(device), loader[2][
                    1].to(device), loader[3][1].to(device)
                labels = torch.cat((labelsXX, labelsXY, labelsYX, labelsYY), 0)
                det_labelsXX = F.one_hot(labelsXX, num_classes=2).to(dtype=torch.float64)
                det_labelsXY = F.one_hot(labelsXY, num_classes=2).to(dtype=torch.float64)
                det_labelsYX = F.one_hot(labelsYX, num_classes=2).to(dtype=torch.float64)
                det_labelsYY = F.one_hot(labelsYY, num_classes=2).to(dtype=torch.float64)
                # det_labels = labels_image_tensors[labels]
                optimizer.zero_grad()  # 梯度清零
                out_label, out_imgXX,out_imgXY,out_imgYX,out_imgYY = model([imagesXX,imagesXY,imagesYX,imagesYY])  # 得到预测各个探测器上的光强分布以及探测层光强分布
                # _, predictedXX = torch.max(out_labelXX.data, 1)
                # _, predictedXY = torch.max(out_labelXY.data, 1)
                # _, predictedYX = torch.max(out_labelYX.data, 1)
                # _, predictedYY = torch.max(out_labelYY.data, 1)
                _, predicted = torch.max(out_label.data, 1)
                correct1 = correct1 + (predicted == labels).sum().item()
                total2 = total2+ labelsXX.size(0) + labelsXY.size(0) + labelsYX.size(0) + labelsYY.size(
                    0)  # 得到一个batch的标签总数
                full_int_imgXX = out_imgXX.sum(axis=(1, 2))
                #print((out_imgXX / full_int_imgXX[:, None, None]).shape,det_labels.shape)
                #loss = loss_function(out_imgXX / full_int_imgXX[:, None, None], det_labelsXX)
                det_labels = torch.cat((det_labelsXX, det_labelsXY, det_labelsYX, det_labelsYY), 0)
                loss = loss_function(out_label, det_labels)
                ep_loss += loss.item()  # 更新本次epoch的损失
        # test_loss_hist.append(ep_loss / len(testloader))
        test_acc_hist.append(correct2 / total2)
        # 如果最后一次训练的准确率大于之前最好的准确率，则将最后一次的模型保存为最佳模型。
        # if test_acc_hist[-1] > best_acc:
        #     best_model = copy.deepcopy(model)
        # print(f"Epoch={epoch} train loss={train_loss_hist[epoch]:.4}, test loss={test_loss_hist[epoch]:.4}")
        print(f"train acc={train_acc_hist[epoch]:.4}, test acc={test_acc_hist[epoch]:.4}")
        print("-----------------------")

    return  train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist

wl = 528e-9
pixel_size = 400e-9
# 定义模型，损失函数和优化器
model = DNN( num_layers =3, wl = wl, pixel_size = pixel_size, distance = 30e-7).to(device)#30um
criterion = torch.nn.MSELoss(reduction='sum').to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
train_loss_hist, train_acc_hist,test_loss_hist, test_acc_hist= train1(model,
                          criterion,optimizer, train_dataloaderXX,train_dataloaderXY,train_dataloaderYX,train_dataloaderYY,test_dataloaderXX,test_dataloaderXY,test_dataloaderYX,test_dataloaderYY, epochs = 4,  device = device)


PADDING = (N_pixels - IMG_SIZE) // 2
def visualize(image, label,index):
    image_padded = F.pad(image, pad=(PADDING, PADDING, PADDING, PADDING))
    # print(image_padded[0])
    image_E = torch.sqrt(image_padded)
    out = model(image_E.to(device),index)
    # print(label,out[1][0])
    # print(torch.mul(labels_image_tensors[label],out[1][0]).sum(dim=(0,1)),out[1][0].sum(dim=(0,1)))
    # print(torch.mul(labels_image_tensors[label],out[1][0]).sum(dim=(0,1))/out[1][0].sum(dim=(0,1)))
    print(torch.mul(labels_image_tensorsXX[label], out[1][0]).sum(dim=(0, 1)), out[1][0].sum(dim=(0, 1)))
    print(torch.mul(labels_image_tensorsXX[label], out[1][0]).sum(dim=(0, 1)) / out[1][0].sum(dim=(0, 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image_E[0], interpolation='none')
    ax1.set_title(f'Input image with\n total intensity {image_padded[0].sum():.2f}')
    output_image = out[1].detach().cpu()[0]
    ax2.imshow(output_image, interpolation='none')
    ax2.set_title(f'Output image with\n total intensity {output_image.sum():.2f}')
    fig.suptitle("label={}".format(label), x=0.51, y=0.85)
    plt.show()
# for i,(image,label) in enumerate(test_dataloaderXX):
#     visualize(image[0],label[0],0)
#     if i==3:
#         break
# for i,(image,label) in enumerate(test_dataloaderXY):
#     visualize(image[0],label[0],1)
#     if i==3:
#         break
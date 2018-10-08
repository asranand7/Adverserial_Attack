#=================================== Import Libraries ==================================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn 
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#=================================== VGG-16 Network =============================================================================================


class VGG16(nn.Module):
  def __init__(self):
    super(VGG16,self).__init__()

    self.block1 = nn.Sequential(
                  nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3, padding =1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.3))

    self.block2 = nn.Sequential(
                  nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size = 3, padding =1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.4))

    self.block3 = nn.Sequential(
                  nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3, padding =1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.4))

    self.block4 = nn.Sequential(
                  nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3, padding =1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2) ,
                  nn.Dropout2d(0.4))

    self.block5 = nn.Sequential(
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3, padding =1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.5) )

    self.fc =     nn.Sequential(
                  nn.Linear(512,100),
                  nn.Dropout(0.5),
                  nn.BatchNorm1d(100),
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  nn.Linear(100,10), )
                  
                  


  def forward(self,x):
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out = self.block4(out)
    out = self.block5(out)
    out = out.view(out.size(0),-1)
    out = self.fc(out)

    return out
# ====================================== Load model and data =====================================================================================
model = VGG16()
if torch.cuda.is_available():
  model.cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(),lr = 0.001,momentum = 0.9,weight_decay = 0.006)
# schedule = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma = 0.7)

state = torch.load('./model_170_85.pth')
model.load_state_dict(state['model'])

test_images = np.load('test_images.npy')   # Load the data
test_label = np.load('test_label.npy')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# =============================== DeepFool function =======================================================================================

def DeepFool(image,model,num_classes = 10,maxiter = 50,min_val = -1,max_val = 1,true_label = 0):
  '''   Our VGG model accepts images with dimension [B,C,H,W] and also we have trained the model with the images normalized with mean and std.
        Therefore the image input to this function is mean ans std normalized.
        min_val and max_val is used to clamp the final image
  '''
  image = torch.from_numpy(image)

  is_cuda = torch.cuda.is_available()
  
  if is_cuda:
    model = model.cuda()
    image = image.cuda()

  model.train(False)
  f = model(Variable(image[None,:,:,:],requires_grad = True)).data.cpu().numpy().flatten()
  I = (np.array(f)).argsort()[::-1]
  label = I[0]  # Image label

  input_shape = image.cpu().numpy().shape

  if(label != true_label): # Find adversarial noise only when Predicted label = True label
    print("Predicted label is not the same as True Label ")
    return np.zeros(input_shape),0,-1,0, np.zeros(input_shape)
  
  
  pert_image = image.cpu().numpy().copy()   # pert_image stores the perturbed image
  w = np.zeros(input_shape)                # 
  r_tot = np.zeros(input_shape)   # r_tot stores the total perturbation
  
  pert_image = torch.from_numpy(pert_image)
  
  if is_cuda:
    pert_image = pert_image.cuda()
    
  x = Variable(pert_image[None,:,:,:],requires_grad = True)
  fs = model(x)
  fs_list = [ fs[0,I[k]] for k in range(num_classes) ]
  
  k_i = label  # k_i stores the label of the ith iteration
  loop_i = 0
  
  
  while loop_i < maxiter and k_i == label:
    pert = np.inf
    fs[0,I[0]].backward(retain_graph = True)  
    grad_khat_x0 = x.grad.data.cpu().numpy()  # Gradient wrt to the predicted label
    
    for k in range(1,num_classes):
      if x.grad is not None:
        x.grad.data.fill_(0)
      fs[0,I[k]].backward(retain_graph = True)
      grad_k = x.grad.data.cpu().numpy()
      
      w_k = grad_k - grad_khat_x0
      f_k = (fs[0,I[k]] - fs[0,I[0]]).data.cpu().numpy()
      
      pert_k = abs(f_k)/(np.linalg.norm(w_k.flatten()))
      
      if pert_k < pert:
        pert = pert_k
        w = w_k

    r_i = (pert)*(w/np.linalg.norm(w.flatten()))
    r_tot = np.float32(r_tot +  r_i.squeeze())

    if is_cuda:
      pert_image += (torch.from_numpy(r_tot)).cuda()
    else:
      pert_image += toch.from_numpy(r_tot)
    
    x = Variable(pert_image,requires_grad = True)
    fs = model(x[None,:,:,:])
    k_i = np.argmax(fs.data.cpu().numpy().flatten())
    
      
    loop_i += 1
  pert_image = torch.clamp(pert_image,min_val,max_val) 
  pert_image = pert_image.data.cpu().numpy()
  
  return r_tot,loop_i,label,k_i,pert_image



# =============================== Visualisation =======================================================================================

mean = np.array([0.485, 0.456, 0.406])  # Mean and std of the data
mean = mean[:,None,None]
std = np.array([0.229, 0.224, 0.225])
std = std[:,None,None]

def show_image(img1,img2,mean,std):
  img1 = img1*std + mean   # unnormalize
  img2 = img2*std + mean
  img1 = img1.clip(0,1)
  img2 = img2.clip(0,1)
  
  img1 = np.transpose(img1,(1,2,0))
  img2 = np.transpose(img2,(1,2,0))
  
  
  noise = img2- img1
  noise = abs(noise)
  noise =noise.clip(0,1)

  disp_im = np.concatenate((img1,img2,2*noise),axis = 1)
  plt.imshow(disp_im)


idx = np.random.randint(0,10000)

sample_image = test_images[idx]
img_label = test_label[idx]
# since the image input to the DeepFool function in normalized according to the above mean and std therefore the min_val and max_val is not {0,1} but {-2.117,2.64}
r_tot,loop_i,label,k_i,pert_image = DeepFool(sample_image,model,num_classes = 10,maxiter = 50,min_val = -2.117,max_val = 2.64,true_label = img_label)

print("Clean Label: " ,classes[label]," Adversarial Label:" ,classes[k_i])
show_image(sample_image,pert_image,mean,std)

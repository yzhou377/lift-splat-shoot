from sklearn.cluster import KMeans
import numpy as np 
import torch 
import torchvision
from PIL import Image
import math 
from torchvision import transforms
from glob import glob
from tensorboardX import SummaryWriter
import os 

class CarlaData(object):
    def __init__(self, root, is_train):
        self.root = root
        self.is_train=is_train
        # load all image files, sorting them to
        # ensure that they are aligned
        self.ixes= self.prepro()

    def prepro(self):
        """
        Return a list of all scenes, each scene is a dictionary with 'label' and 'inputs' for 6 cameras
        Incomplete data are elimited here.  
        DONE: scenes are incomplete on Mar.08, from Takeda in email. YZ. 
        """
        data_summary=[]
        if self.is_train:
            start_folder=0
            end_folder=69
        else:
            start_folder=70
            end_folder=99

        input_folder= sorted(glob(os.path.join(self.root, "preprocessed",'bev_rgb_acc_inpaint-*')))
        for input_image_dir in input_folder:
            label_folder_index= input_image_dir[-17:-11]
            label_file_index= input_image_dir[-10:-4]
            data_summary.append({'label': os.path.join(self.root,label_folder_index,"bev_keypoint_{}.png".format(label_file_index) ), 'input_dir':input_image_dir})
        return data_summary

    def get_keypoints(self,rec):
        keypts_path= rec['label']
        green_value= (0,255,255)
        red_value= (255,0,255)


        image=Image.open(keypts_path)
        raw_pixel= np.asarray(image)
        green_pixels= np.vstack(np.where(np.all(raw_pixel==green_value,axis=-1)==True)) 
        red_pixels= np.vstack(np.where(np.all(raw_pixel==red_value,axis=-1)==True))
        g_pts_count= int(green_pixels.shape[1]/65)
        r_pts_count= int(red_pixels.shape[1]/65)

        if g_pts_count==0 or r_pts_count==0:
            return np.zeros([14])

        g_fit=KMeans(n_clusters=g_pts_count, random_state=0).fit(green_pixels.T) 
        r_fit=KMeans(n_clusters=r_pts_count, random_state=0).fit(red_pixels.T)

        g_pts= g_fit.cluster_centers_
        r_pts= r_fit.cluster_centers_
        all_pts= np.vstack((g_pts,r_pts))
        
        #center:
        center= np.mean(all_pts,axis=0).reshape(2,1)
        #up:
        up_pts= all_pts[np.where( (all_pts[:,0]<all_pts[:,1]) * (all_pts[:,0]<1000- all_pts[:,1]))]
        if up_pts.shape[0]==0:
            up_num=np.array([[0]])
            up_center= np.array([[0],[0]])
        else:
            up_num= up_pts.shape[0]
            up_center= np.mean(up_pts,axis=0,dtype=int).reshape(2,1)

        #bottom:
        bottom_pts= all_pts[np.where( (all_pts[:,0]>all_pts[:,1]) * (all_pts[:,0]>1000- all_pts[:,1]))]
        if bottom_pts.shape[0]==0:
            bottom_num=np.array([[0]])
            bottom_center= np.array([[0],[0]])
        else:
            bottom_num= bottom_pts.shape[0]
            bottom_center= np.mean(bottom_pts,axis=0,dtype=int).reshape(2,1)

        #left:
        left_pts= all_pts[np.where( (all_pts[:,0]>all_pts[:,1]) * (all_pts[:,0]<1000- all_pts[:,1]))]
        if left_pts.shape[0]==0:
            left_num=np.array([[0]])
            left_center= np.array([[0],[0]])
        else:
            left_num= left_pts.shape[0]
            left_center= np.mean(left_pts,axis=0,dtype=int).reshape(2,1)

        #right:
        right_pts= all_pts[np.where( (all_pts[:,0]<all_pts[:,1]) * (all_pts[:,0]>1000- all_pts[:,1]))]
        if right_pts.shape[0]==0:
            right_num=np.array([[0]])
            right_center= np.array([[0],[0]])
        else:
            right_num= right_pts.shape[0]
            right_center= np.mean(right_pts,axis=0,dtype=int).reshape(2,1)

        #Returning a 14 dimensional vector 
        return np.vstack((center, up_num, up_center, bottom_num, bottom_center, left_num, left_center, right_num, right_center )).reshape(14,)


    def get_image_data(self, rec):
        """
        Return the image normalized by tools.normalize
        """
        def downsize(img,rate):
            img = img.resize((int(img.size[0]*rate),int(img.size[1]*rate)),Image.ANTIALIAS)
            return  img

        input_image = Image.open(rec["input_dir"])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocess_raw= transforms.Compose([transforms.ToTensor()])
        input_tensor = preprocess(downsize(input_image,1/8))
        return preprocess_raw(input_image), input_tensor

    def __getitem__(self, idx):
        rec = self.ixes[idx]
        input_image, input_tensor = self.get_image_data(rec)
        target_tensor= torch.Tensor(self.get_keypoints(rec))
        #if self.transforms is not None:
        #    img, target = self.transforms(img, target)

        return input_image, input_tensor, target_tensor

    def __len__(self):
        return len(self.ixes)


def compile_data(dataroot, bsz, nworkers):

    traindata= CarlaData(root= dataroot, is_train= True)
    valdata= CarlaData(root= dataroot, is_train=False)
    
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                            shuffle=True,
                                            num_workers=nworkers,
                                            drop_last=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    return trainloader, valloader 

def tensorboard_visualiza(model, writer, dataloader, is_train, device):
    sampeld_image_data= iter(dataloader)
    input_image, input_tensor, target= sampeld_image_data.next()
    #print(rec['label'])
    # Display the input image set 
    
    first_image= list(torch.unbind(input_image[0]))
    input_tensor
    first_label= target[0].numpy()


    print('Yes')
    img_grid= torchvision.utils.make_grid(tensor= first_image, nrow=1)
    if is_train:
        writer.add_image('train_input', img_grid)
    else: 
        writer.add_image('val_input',img_grid)

    # Display GT Label Image
        #TODO
    # Display Predict Label Image
        #TODO


if __name__ == '__main__':

     # The data preparation
     data_root= "/media/m/377445e5-f724-4791-861c-730d2b0bba3f/carla_map_bev_10k_v2"
     logdir= "/home/m/lift-splat-shoot/src/image_regression/log" 

     bsz=6
     nworkers=10            
     lr=1e-3
     weight_decay=1e-7
     gpuid=0
     num_epochs=5
     counter = 0

     trainloader, valloader = compile_data(data_root, bsz=bsz, nworkers=nworkers)
    
     device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

     model= torchvision.models.resnet18(pretrained=True)
     model.fc= torch.nn.Linear(512,14)
     model.to(device)
     opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
     loss_fn = torch.nn.MSELoss()

     writer = SummaryWriter(logdir=logdir)


     model.train()
     for epoch in range(num_epochs):
         for batch_idx, (input_image, data, target) in enumerate(trainloader):
             data, target = data.to(device), target.to(device)
             opt.zero_grad()
             output = model(data)
             loss = loss_fn(output, target)
             loss.backward()
             opt.step()
             counter += 1

             if counter % 10 == 0:
                 print(counter, loss.item())
                 writer.add_scalar('train/loss', loss, counter)

             if counter % 1 == 0:
                tensorboard_visualiza(model= model, writer= writer, dataloader= trainloader, is_train=1, device= device)


     print("That's it!")



import os
import cv2
import torch
import torchvision.transforms as transforms


Color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
        [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 
              'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  
              'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def nms(bboxes, scores, threshold=0.5):
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
        _, order = scores.sort(0, descending=True)    # 降序排列

        keep = []
        while order.numel() > 0:       # torch.numel()返回张量元素个数
            if order.numel() == 1:     # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()    # 保留scores最大的那个框box[i]
                keep.append(i)

            # 计算box[i]与其余各框的IOU(思路很好)
            xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

            iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
            idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
            if idx.numel() == 0:
                break
            order = order[idx+1]  # 修补索引之间的差值
        return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor

def decoder(pred):
    '''
    pred (tensor) 1x7x7x26
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 7
    boxes=[] # [tensor([[x1,y1,x2,y2]]),tensor([[x1,y1,x2,y2]])...]
    cls_indexs=[] # [tensor(class_index, device='cuda:0'), tensor(13, device='cuda:0')...]
    probs = [] # [tensor([0.4422]), tensor([0.0885]...]
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #torch.Size([7, 7, 26])
    #print('pred:',pred.size())
    contain1 = pred[:,:,4].unsqueeze(2) #torch.Size([7, 7, 1]) 每個grid裡第一個bbx的confidence score
    #print('contain1:',contain1.size())
    contain2 = pred[:,:,9].unsqueeze(2) #torch.Size([7, 7, 1]) 每個grid裡第二個bbx的confidence score
    #print('contain2:',contain2)
    contain = torch.cat((contain1,contain2),2) #torch.Size([7, 7, 2]) 每個grid裡 [confidence score_1, confidence score_2]
    #print(contain.size())
    mask1 = contain > 0.02 #torch.Size([7, 7, 2]) 大於threshold_1
    #print(mask1.size())
    mask2 = (contain==contain.max()) #torch.Size([7, 7, 2]) we always select the best contain_prob what ever it >0.9?
    #print(mask2)
    #print('mask1+mask2:',mask1+mask2)
    mask = (mask1+mask2).gt(0)
    #print('mask:',mask)
    #print(mask)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2): #bbx_index
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                #if mask[i,j,b] == 1: #check gt_bbx
                if mask[i,j,b] > 0.02: #有過threshold的grid的box
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4] #x,y,w,h
                    #print('box:',box) #box: tensor([0.4219, 0.5938, 0.0580, 0.0603]) 
                    
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]]) #該box的confidence score 
                    xy = torch.FloatTensor([i,j])*cell_size #cell左上角  up left of cell [0~1,0~1]
                    #i,j: 5, 6
                    #print('xy:',xy)
                    #xy = 0.714, 0.8571
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image [0~1,0~1]
                    #print('box[:2]',box[:2])
                    #0.060, 0.084 + 0.714, 0.8571 = 0.774, 0.9411
                    box_xy = torch.FloatTensor(box.size())# convert[cx,cy,w,h] to [x1,y1,x2,y2]:torch.Size([4])
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    #0.774, 0.941 - 0.029, 0.0301 = 0.745,0.9109   (* 512 -> 381,466)
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    #0.774, 0.941 + 0.029, 0.0301 = 0.803,0.9711   (* 512 -> 411,497)
                    #print('box_xy',box_xy * 512)
                    # [381,466,411,497]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0) #torch.max(input, dim=0) 找每個column最大的值
                    #return max, argmax -> 找16class中機率最大的class
                    #ex: tensor(0.8114, device='cuda:0'), tensor(13, device='cuda:0')
                    #print(max_prob,cls_index)
                    
                    if float((contain_prob*max_prob)[0]) > 0.11:  #0.4
                        boxes.append(box_xy.view(1,4)) #torch.Size([1,4])
                        cls_indexs.append(cls_index.view(1,1)) #to be able to cat later
                        probs.append(contain_prob*max_prob)
                        
    if len(boxes) == 0: #此image沒有pred出box
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #torch.Size([n,4]) 往row的方向下疊n個row (n個boxes的x1,y1,x2,y2) 有猛到（因為原本是list）
        #print('boxes after cat:', boxes)
        probs = torch.cat(probs,0) #(n,)
        #print('probs after cat:', probs)
        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]  #tensor([433.5357, 375.2590, 461.8572, 404.6696]),


def check_labels(image, label, save_path):
    '''
    @image torch.Size([3, 448, 448])
    @label torch.Size([7, 7, 26]) -> tensor([row,col,label]) -> label = [x,y,w,h,c,0,0,0,0,one-hot(16-class)]
    print the bbox on the cooresponding image
    '''
    #opencv        （height,width,channels）
    #deep learning （channels,height,width）
    img = image
    img = img.numpy()
    img = img.transpose(1,2,0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img *= 255
    _,w,h = image.shape
    #print(image.shape)
    #print(w,h)
    #print(type(img))
    #print(img)
    
    boxes,cls_indexs,probs = decoder(label.unsqueeze(0).to(dtype=torch.float32))
    #print('pred.size',pred.size())
    #print('pred.dtype',pred.dtype)
    result = []

    for i,box in enumerate(boxes):
        
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),classnames[cls_index],prob])
        #print(box)
    print('result:',result)
            
    for left_up,right_bottom,class_name,prob in result:
        color = Color[classnames.index(class_name)]
        cv2.rectangle(img,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
    
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite(os.path.join(save_path, 'result_check_gt.jpg'),img)
    
    
def predict_gpu(model, image_name, device):#,root_path='hw2_train_val/val1500/'
 
    result = []
    image = cv2.imread(image_name)#root_path+
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #mean = (123,117,104)#RGB
    #img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        img = img[None,:,:,:]
        img = img.to(device)

        pred = model(img) #1x7x7x26
        pred = pred.cpu()
        boxes,cls_indexs,probs = decoder(pred)
        print('pred.size',pred.size())
        print('pred.dtype',pred.dtype)

        for i,box in enumerate(boxes):
            
            x1 = int(box[0]*w)
            x2 = int(box[2]*w)
            y1 = int(box[1]*h)
            y2 = int(box[3]*h)
            
            cls_index = cls_indexs[i]
            cls_index = int(cls_index) # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            result.append([(x1,y1),(x2,y2),classnames[cls_index],image_name,prob])
    return result
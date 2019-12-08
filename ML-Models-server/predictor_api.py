import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from utils import CTCLabelConverter, AttnLabelConverter # local
from dataset import RawDataset, AlignCollate # local
from model import Model # local
import sys
import requests
from googletrans import Translator
import os
import time
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import cv2
from skimage import io
import numpy as np
import craft_utils # local
import imgproc # local
import file_utils # local
import json
import zipfile
from craft import CRAFT # local 
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = argparse.Namespace(canvas_size=1280, cuda=False, link_threshold=0.4, low_text=0.4, mag_ratio=1.5, poly=False, refine=False, refiner_model='weights/craft_refiner_CTW1500.pth', show_time=False, test_folder='images', text_threshold=0.7, trained_model='craft_mlt_25k.pth')

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



def make_prediction(url):
    # craft net run
    with open('FB_IMG_1490534565948.jpg', 'wb') as f:
        f.write(requests.get(url).content)

    input_image = ['FB_IMG_1490534565948.jpg']
    print(input_image)
    CraftNetOut = runCraftNet(input_image) # this is a list, index 0 is the marked image, index 1 is the text coords
    img = Image.fromarray(CraftNetOut[0])
    text = CraftNetOut[1]

    # segment images according to text coords 
    segmentedImages = []
    for coordinateRow in text:
        valuesList = coordinateRow.split(",")
        x_values = []
        y_values = []

        for x in range(0,8,2):
            x_values.append(int(valuesList[x]))

        for x in range(1,8,2):
            y_values.append(int(valuesList[x]))

        img2 = img.crop((min(x_values), min(y_values), max(x_values), max(y_values)))
        segmentedImages.append(img2)
        # print(segmentedImages)

    # deep text run
    finalOut = runDeepTextNet(segmentedImages)
    finalOut = translate(finalOut)
    print(finalOut)
    return finalOut
    # output

def runCraftNet(image_list): # image list is the folder containing the images

    args = argparse.Namespace(canvas_size=1280, cuda=False, link_threshold=0.4, low_text=0.4, mag_ratio=1.5, poly=False, refine=False, refiner_model='weights/craft_refiner_CTW1500.pth', show_time=False, test_folder='images', text_threshold=0.7, trained_model='craft_mlt_25k.pth')
    net = CRAFT()     # initialize
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    net.eval()

    # image_list, _, _ = file_utils.get_files(args.test_folder)
    t = time.time()
    # result_folder = './result/'

    # load data
    refine_net = None

    for k, image_path in enumerate(image_list):
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

    # print("elapsed time : {}s ".format(time.time() - t))
    img = np.array(image[:,:,::-1])
    txt = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        strResult = ','.join([str(p) for p in poly])
        txt.append(strResult)
    
    return [img, txt]

# img = runCraftNet()[0]
# text = runCraftNet()[1]

# img = Image.fromarray(img)

# segmentedImages = []
# for coordinateRow in text:
#     valuesList = coordinateRow.split(",")
#     x_values = []
#     y_values = []

#     for x in range(0,8,2):
#         x_values.append(int(valuesList[x]))

#     for x in range(1,8,2):
#         y_values.append(int(valuesList[x]))

#     # a 4-tuple defining the left, upper, right, and lower pixel coordinate.
#     img2 = img.crop((min(x_values), min(y_values), max(x_values), max(y_values)))
#     segmentedImages.append(img2)

######## 2nd model

def translate(text):
    translator = Translator()
    translations = translator.translate(text, dest = 'en')
    for translation in translations:
        return translation.text

def runDeepTextNet(segmentedImagesList):
    opt = argparse.Namespace(FeatureExtraction='ResNet', PAD=False, Prediction='Attn', 
    SequenceModeling='BiLSTM', Transformation='TPS', batch_max_length=25, batch_size=192, 
    character='0123456789abcdefghijklmnopqrstuvwxyz', hidden_size=256, 
    image_folder='demo_image/', imgH=32, imgW=100, input_channel=1, num_class=38,
        num_fiducial=20, num_gpu=0, output_channel=512, rgb=False, 
            saved_model='TPS-ResNet-BiLSTM-Attn.pth', sensitive=False, workers=4)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to('cpu')
    directory = "TPS-ResNet-BiLSTM-Attn.pth"
    model.load_state_dict(torch.load(directory, map_location='cpu'))

    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=segmentedImagesList, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()

    out_preds_texts = []
    for image_tensors, image_path_list in demo_loader:
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        preds = model(image, text_for_pred, is_train=False)
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            # print(pred)
            out_preds_texts.append(pred)
    # print(out_preds_texts)

    sentence_out = [' '.join(out_preds_texts)]
    return(sentence_out)

print("")
# deepNetout = runDeepTextNet()
# image_list, _, _ = file_utils.get_files(args.test_folder)
# print(image_list)
# img = Image.open()
# print(make_prediction(image_list))






























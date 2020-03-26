import sys
import tools_matrix as tools
import caffe,time
import cv2,os
import numpy as np
#caffe.set_mode_gpu()
deploy = 'onet/48net.prototxt'
caffemodel = 'onet/48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)


class SSDDetection:
    def __init__(self,model_def, model_weights):
        self.net = caffe.Net(model_def,model_weights,caffe.TEST)
        self.height = self.net.blobs['data'].shape[2]
        self.width = self.net.blobs['data'].shape[3]

    def preprocess(self,src):
        img = cv2.resize(src, (self.height,self.width))
        img = np.array(img, dtype=np.float32)
        img -= np.array((127.5, 127.5, 127.5)) 
        return img

    def detect(self,img, conf_thresh=0.2, topn=10):
        height,width,c=img.shape
        self.net.blobs['data'].data[...] = self.preprocess(img).transpose((2, 0, 1)) 
        detections = self.net.forward()['detection_out']
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i]
            ymin = top_ymin[i] 
            xmax = top_xmax[i] 
            ymax = top_ymax[i] 
            score = top_conf[i]
            label = int(top_label_indices[i])
            if label:
               result.append([max(xmin* width,0), max(ymin* height,0), max((xmax-0)* width,0), max((ymax-0)* height,0), score])
        return result


def detectFace(img,threshold,rectangles):
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    if len(rectangles)==0:
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
    return rectangles


def testdir(dir,detection):
    files=os.listdir(dir)
    for file in files:
        imgpath=dir+"/"+file
        print file
        img = cv2.imread(imgpath)
        start=time.time()
        results=detection.detect(img)
        end=time.time()
        cost="%0.2fms" %((end-start)*1000)
        print cost
        threshold = [0.6,0.6,0.7]
        rectangles = detectFace(img,threshold,results)
        draw = img.copy()
        for rectangle in rectangles:
            cv2.putText(draw,"face",(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
            for i in range(5,15,2):
    	        cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),4,(0,255,0))
        cv2.imshow('test',draw)
        k = cv2.waitKey(0) & 0xff
        cv2.destroyAllWindows()
        if k == 27 : return False

def testcamera(detection):
    cap=cv2.VideoCapture(0)
    while True:
        ret,img=cap.read()
        if not ret:
           break
        start=time.time()
        results=detection.detect(img)
        end=time.time()
        threshold = [0.6,0.6,0.7]
        rectangles = detectFace(img,threshold,results)        
        draw = img.copy()
        for rectangle in rectangles:
            cv2.putText(draw,"face",(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
            for i in range(5,15,2):
    	        cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),4,(0,255,0))       
        cost="%0.2fms" %((end-start)*1000)
        print cost
        cv2.imshow('test',draw)
        cv2.waitKey(1)

def main():
    model_def = 'tssd/tssd.prototxt'
    model_weights='tssd/tssd.caffemodel'
    detection = SSDDetection(model_def, model_weights)
    testdir('images',detection)
    #testcamera(detection)
main()


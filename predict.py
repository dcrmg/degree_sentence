# -*- coding: utf-8 -*-
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.layers import Dense
from keras.models import Model
import numpy as np
from PIL import Image
import cv2

from keras.optimizers import SGD


def load():
    vgg = VGG16(weights=None,input_shape=(224,224,3))
    ##修改输出层 3个输出
    x  = vgg.layers[-2].output   #获取倒数第二层的输出赋给x
    predictions_class = Dense(4, activation='softmax', name='predictions_class')(x)  #定义一个输出维度是4,激活函数使用softmax的全连接层
    prediction = [predictions_class]
    model = Model(inputs=vgg.input, outputs=prediction) # 创建一个网络模型（model），包含从输入到输出的所有层
    sgd = SGD(lr=0.00001, momentum=0.9)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  #dispensable in prediction only
    model.load_weights('./model/model01.h5')
    return model

##加载模型
model = load()

def predict(path=None,img=None):
    """
    图片文字方向预测
    """
    ROTATE = [0,90,180,270]
    if path is not None:
       im = Image.open(path).convert('RGB')
    elif img is not None:
       im = Image.fromarray(img).convert('RGB')
    w,h = im.size
    xmin,ymin,xmax,ymax = int(0.1*w),int(0.1*h),w-int(0.1*w),h-int(0.1*h)
    im = im.crop((xmin,ymin,xmax,ymax))##剪切图片边缘，清除边缘噪声
    im = im.resize((224,224))
    img = np.array(im)

    img = preprocess_input(img.astype(np.float32))
    pred = model.predict(np.array([img]))
    index = np.argmax(pred,axis=1)[0]
    return ROTATE[index]


if __name__ == '__main__':
    img = cv2.imread('./img/01.jpg')
    angle = predict(img=np.copy(img))
    print '********************'
    print angle
    print '********************'

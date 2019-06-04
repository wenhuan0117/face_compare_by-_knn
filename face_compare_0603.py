import tensorflow as tf
import cv2
import numpy as np
##N_CLASSES=20
import readimages as rd
import random
import time

t_start=time.time()
learning_rate=0.1
num_steps=20
batch_size=100
disp_step=2
dropout=0.75

N_CLASSES=20

train_data_path='E:/phython/opencv/test1/lfw'
txtpath='test.txt'
def load_image(image_path):
    
    img=cv2.imread(image_path)
    img=img/255.0
    image=cv2.resize(img,(224,224))
    xs=[]
    xs.append(image)
    return xs

def creat_train_data(data_path):
    
    imagepaths,labels=rd.read_images(data_path)
    x_data,y_data,ran_list=rd.load_images(batch_size,imagepaths,labels)

    with open(txtpath,"w") as f:
        for i in ran_list:
            f.write(imagepaths[i]+'  ')
            f.write(imagepaths[i].split('\\')[-2]+'  ')
            f.write(str(labels[i]))
            f.write('\n')
    return x_data,y_data

x_data,y_data=creat_train_data(train_data_path)

def anc_p_n_data(txtpath):
    with open(txtpath,'r') as f:
        data=f.readlines()
        imgdir={}
        for eachline in data:
            imgkey=eachline.split(' ')[0]
            imgvalue=int(eachline.split(' ')[-1])
            imgdir[imgkey]=imgvalue
            
    anc=[]
    ap=[]
    an=[]
    for key in imgdir:

        for key2 in imgdir:
            if (imgdir[key2]==imgdir[key])and(key2!=key):            
                break
                    
        for key3 in imgdir:
            if (imgdir[key3]==imgdir[key]):
                pass
            else:
                if (imgdir[key2]==imgdir[key]):
                    anc.append(key)
                    ap.append(key2)
                    an.append(key3)
    with open('anc.txt','w') as f:
        for i in anc:
            f.write(i+'\n')
    with open('ap.txt','w') as f:
        for i in ap:
            f.write(i+'\n')
    with open('an.txt','w') as f:
        for i in an:
            f.write(i+'\n')
    return anc,ap,an

anc,ap,an=anc_p_n_data(txtpath)

def build_vgg16(tfx):
    
    data_dict=np.load(npy_path,encoding='latin1').item()
    vgg_mean=[103.939,116.779,123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=tfx * 255.0)
    bgr = tf.concat(axis=3, values=[
        blue -vgg_mean[0],
        green - vgg_mean[1],
        red -vgg_mean[2],
    ])

    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
            return lout
    conv1_1 =conv_layer(bgr, "conv1_1")
    conv1_2 =conv_layer(conv1_1, "conv1_2")
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 =conv_layer(pool1, "conv2_1")
    conv2_2 =conv_layer(conv2_1, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 =conv_layer(pool2, "conv3_1")
    conv3_2 =conv_layer(conv3_1, "conv3_2")
    conv3_3 =conv_layer(conv3_2, "conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 =conv_layer(pool3, "conv4_1")
    conv4_2 =conv_layer(conv4_1, "conv4_2")
    conv4_3 =conv_layer(conv4_2, "conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 =conv_layer(pool4, "conv5_1")
    conv5_2 =conv_layer(conv5_1, "conv5_2")
    conv5_3 =conv_layer(conv5_2, "conv5_3")
    pool5 = max_pool(conv5_3, 'pool5')

    flatten = tf.reshape(pool5, [-1, 7*7*512])
    fc6 = tf.layers.dense( flatten, 50, tf.nn.relu, name='fc6',reuse=tf.AUTO_REUSE)
    return fc6

npy_path='E:/phython/tensorflow-test/transfer learning/vgg16.npy'


x_data1=[]
x_data2=[]
x_data3=[]


for k in range(0,586,10):
    x_data1.append(load_image(anc[k])[0])
    x_data2.append(load_image(ap[k])[0])
    x_data3.append(load_image(an[k])[0])
    
length=len(x_data1)
length_list=list()
for i in range(length):
    length_list.append(i)

    

##x_data1=np.array(x_data1)
##x_data2=np.array(x_data2)
##x_data3=np.array(x_data3)

tfx=tf.placeholder(tf.float32,[None,224,224,3])
tfy=tf.placeholder(tf.float32,[None,224,224,3])
tfz=tf.placeholder(tf.float32,[None,224,224,3])

out1=build_vgg16(tfx)
out2=build_vgg16(tfy)
out3=build_vgg16(tfz)

l1=tf.reduce_sum(tf.square(out1-out2),axis=1)
l2=tf.reduce_sum(tf.square(out1-out3),axis=1)
l3=tf.reduce_max(l1)
l4=tf.reduce_min(l2)
loss=tf.div(l1,l2)

global_step=tf.Variable(0,trainable=False)
lr=tf.train.exponential_decay(learning_rate,global_step,20,0.99)
train_op=tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

save_model_path='./facecompare/mynet'
def train_model():
    with tf.Session() as sess:
        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            ran_li=random.sample(length_list,5)
            x_batch1=[]
            x_batch2=[]
            x_batch3=[]
            for j in ran_li:
                x_batch1.append(x_data1[j])
                x_batch2.append(x_data2[j])
                x_batch3.append(x_data3[j])

            feed_dict={tfx:x_batch1,tfy:x_batch2,tfz:x_batch3}
            sess.run(train_op,feed_dict)
            if i%disp_step==0:
                os1,os2=sess.run([l1,l2],feed_dict)
                print(os1,os2)
        saver.save(sess,save_model_path)
    
        
img_path1='E:/phython/opencv/test1/lfw\\Laura_Linney\\Laura_Linney_0001.jpg'
img_path2='E:/phython/opencv/test1/lfw\\Laura_Linney\\Laura_Linney_0002.jpg'

def test_model(img_path1,img_path2):        
    test_img1=load_image(img_path1)
    test_img2=load_image(img_path2)
    test_img1=np.array(test_img1,dtype=np.float32)
    test_img2=np.array(test_img2,dtype=np.float32)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,save_model_path)
        test_out1=build_vgg16(test_img1)
        test_out2=build_vgg16(test_img2)
        test_ol=tf.reduce_sum(tf.square(test_out1-test_out2),axis=1)
        test_ol1=sess.run(test_ol)
        
    print('oushijuli:',test_ol1)

##  train or use?  
train_model()
t_end=time.time()
print("train time:",(t_end-t_start))
test_model(img_path1,img_path2)

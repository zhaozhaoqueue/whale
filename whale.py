import numpy as np
import pandas as pd
import os
import tensorflow as tf
import skimage.io
import skimage.transform
import imgaug              # For creating distorted images
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def one_hot():
    df = pd.read_csv("train.csv")
    labels = df["Id"]
    uniques, ids = np.unique(labels, return_inverse=True)
    encoded_labels = np_utils.to_categorical(ids, len(uniques))
    return encoded_labels, uniques

def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # # we crop image from center
    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    # resized_img = skimage.transform.resize(img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    resized_img = skimage.transform.resize(img, (224, 224, 3))   # shape [224, 224, 3]
    return resized_img

def load_train_data():
    df = pd.read_csv("train.csv")
    dir = "./train"
    imgs = []
    # for file in os.listdir(dir):
    #     if not file.lower.endswith('.jpg'):
    #         continue
    #     if file not in df["Image"]:
    #         continue
    #     try:
    #         resized_img = load_img(os.path.join(dir, file))
    #     except OSError:
    #         continue
    #     df.loc[df["Image"] == file, ["img"]] = resized_img
    dir_list = os.listdir(dir)
    for _, row in df.iterrows():
        if row["Image"] not in dir_list:
            imgs.append(None)
            continue
        try:
            resized_img = load_img(os.path.join(dir, row["Image"]))
        except OSError:
            imgs.append(None)
        imgs.append(resized_img)
    # # One-hot encoding id
    # labels, org_laldic = one_hot(df["Id"])
    labels, _ = one_hot()
    # return df["Id"], imgs
    return np.array(imgs), np.array(labels)

def to_batches(X, y, batch_size=128, seed=1):
    m = X.shape[0]
    # m = len(X)
    num_batches = int(m/batch_size)
    batches = []
    np.random.seed(seed)
    permutation = np.random.permutation(m)
    X_shuffled = np.take(X, permutation, axis=0)
    y_shuffled = np.take(y, permutation, axis=0)
    for i in range(num_batches):
        bt_X = X_shuffled[i*batch_size:(i+1)*batch_size]
        bt_y = y_shuffled[i*batch_size:(i+1)*batch_size]
        batches.append((bt_X, bt_y))
    if(m%batch_size !=0):
        bt_X = X_shuffled[num_batches*batch_size: ]
        bt_y = y_shuffled[num_batches*batch_size: ]
    return batches

# VGG-16 net
# reference: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py
# reference: https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/407_transfer_learning.py
class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        # self.tfy = tf.placeholder(tf.float32, [None, 1])
        # There are 5005 types whales including new_whale
        self.tfy = tf.placeholder(tf.float32, [None, 5005])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 10240, tf.nn.relu, name='fc6')
        # self.out = tf.layers.dense(self.fc6, 1, name='out')
        self.fc7 = tf.layers.dense(self.fc6, 5005, tf.nn.relu, name="fc7")
        self.out = tf.nn.softmax(self.fc7)

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            # self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc7, labels=self.tfy))
            # self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            # Swtich to adam optimizer and decrease learning rate to 1e-4
            self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss

    def predict(self, path):
        x = load_img(path)
        pred_vec = self.sess.run(self.out, {self.tfx:x})
        pred = argmax(pred_vec)
        return pred

    def save(self, path='myVariables'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)

def train(imgs, labels):
    # imgs, labels = load_train_data()
    vgg = Vgg16(vgg16_npy_path="vgg16.npy")
    print("Net built")

    # A previous loss used to stop iteration when the difference of loss is less
    # than a certain number
    prev_loss = 0
    # Train the self-built layers (last 2 layers)
    seed = 0
    for i in range(30):
        seed += 1
        # Batch
        batches = to_batches(imgs, labels, batch_size=512, seed=seed)
        print("Batches type and length: ", type(batches), len(batches))
        train_loss = 0
        for batch in batches:
            mini_X, mini_y = batch
            # print("X type and size: ", type(mini_X), mini_X.shape)
            # print("y type and size: ", type(mini_y), mini_y.shape)
            mini_loss = vgg.train(mini_X, mini_y)
            train_loss += mini_loss
        print(i, "train loss", train_loss)
        if(abs(train_loss - prev_loss) <= 0.1):
            print("At", i, "iteration stops")
            break
        else:
            prev_loss = train_loss

    vgg.save()

# Assume input is list of image name
def predict(images, dir="./train"):
    vgg = Vgg16(vgg16_npy_path="vgg16.npy", restore_from="myVariables")
    _, org_labels = one_hot()
    preds = []
    for img in images:
        # resized_img = load_img(os.path.join(dir, img))
        path = os.path.join(dir, img)
        pred_ind = vgg.predict(path)
        # preds.append((img, pred_ind))
        preds.append(org_labels[pred_ind])
    return preds

# # How to construct validation since unbalance
# def split_train_val():
#     imgs, labels = load_train_data()
#     train_imgs, val_imgs, train_lab, val_lab = train_test_split(imgs, labels, test_size=0.2, random_state=2)
#     return train_imgs, val_imgs, train_lab, val_lab

# Write later
# def to_submit(dir)


if __name__ == "__main__":
    # train_imgs, val_imgs, train_lab, val_lab = split_train_val()
    # train(train_imgs, train_lab)
    imgs, labels = load_train_data()
    train(imgs, labels)

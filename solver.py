import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
import glob
import cv2
from numpy import array
from PIL import Image


def img_read(image_dir):
    cv_img = []
    for img in glob.glob(image_dir):
        n = cv2.imread(img)
        cv_img.append(n)

    return cv_img

def img_read_partial(image_dir, i, bt_size):
    cv_img = []
    for img in glob.glob(image_dir+'/*.jpg')[i*bt_size:(i+1)*bt_size]:
        n = cv2.imread(img)
        cv_img.append(n)

    return cv_img


def resize_images(images, size=[32, 32, 3]):
    # convert float type to integer
    resized_image_arrays = np.zeros([len(images)] + size)
    for i, image in enumerate(images):
        resized_image = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
        # print type(resized_image)
        resized_image_arrays[i] = resized_image

    return resized_image_arrays


class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100, 
                 classical_dir='classical1', metal_rock_dir='metal1', log_dir='logs', sample_save_path='sample',
                 model_save_path='model', pretrained_model='model/classical_model-20000', test_model='model/dtn-1000'):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.classical_dir = classical_dir
        self.metal_rock_dir = metal_rock_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True

    def load_classical(self, image_dir, split='train'):
        print ('loading classical image dataset..')
        '''
        if self.model.mode == 'pretrain':
            image_file = 'extra_32x32.mat' if split=='train' else 'test_32x32.mat'
        else:
            image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        classical = scipy.io.loadmat(image_dir)
        images = np.transpose(classical['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = classical['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        '''
        image_file = '/*.jpg'
        image_dir = image_dir + image_file
        images = img_read(image_dir)
        images = resize_images(images)
        # print images.shape
        images = images / 127.5 - 1
        print ('finished loading classical image dataset..!')
        # print (images[0])
        return images
        # return images, labels

    def load_metal_rock(self, image_dir, split='train'):
        print ('loading metal_rock image dataset..')
        '''
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            metal_rock = pickle.load(f)
        images = metal_rock['X'] / 127.5 - 1
        labels = metal_rock['y']
        '''
        image_file = '/*.jpg'
        image_dir = image_dir + image_file
        images = img_read(image_dir)

        images = np.asarray(images)
        images = resize_images(images)

        images = images / 127.5 - 1

        print ('finished loading metal_rock image dataset..!')
        return images
        # return images, labels

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))+1
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        return merged

    def pretrain(self):
        # load classical dataset
        train_images, train_labels = self.load_classical(self.classical_dir, split='train')
        test_images, test_labels = self.load_classical(self.classical_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
        
        with tf.Session(config=self.config) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter+1):
                i = step % int(train_images.shape[0] / self.batch_size)
                batch_images = train_images[i*self.batch_size:(i+1)*self.batch_size]
                batch_labels = train_labels[i*self.batch_size:(i+1)*self.batch_size] 
                feed_dict = {model.images: batch_images, model.labels: batch_labels}
                sess.run(model.train_op, feed_dict) 

                if (step+1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss], 
                                           feed_dict={model.images: test_images[rand_idxs], 
                                                      model.labels: test_labels[rand_idxs]})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                               %(step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:  
                    saver.save(sess, os.path.join(self.model_save_path, 'classical_model'), global_step=step+1) 
                    print ('classical_model-%d saved..!' %(step+1))

    def train(self):
        # load classical dataset
        classical_images = self.load_classical(self.classical_dir, split='train')
        metal_rock_images = self.load_metal_rock(self.metal_rock_dir, split='train')
        # print classical_images.shape[0], metal_rock_images.shape[0]

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.initialize_all_variables().run()
            # restore variables of F
            '''
            print ('loading pretrained model F..')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            '''
            saver = tf.train.Saver()
            

            print ('start training..!')
            f_interval = 15
            for step in range(self.train_iter+1):
                
                i = step % int(classical_images.shape[0] / self.batch_size)
                # train the model for source domain S
                src_images = classical_images[i*self.batch_size:(i+1)*self.batch_size]
                # i = step % int(10245 / self.batch_size)
                # src_images = img_read_partial(self.classical_dir, i, self.batch_size)
                # src_images = resize_images(src_images)
                # src_images = src_images / 127.5 - 1
                
                feed_dict = {model.src_images: src_images}
                
                sess.run(model.d_train_op_src, feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                
                if step > 1600:
                    f_interval = 30
                
                if i % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)
                
                if (step+1) % 10 == 0:
                    print ('source - step : ', step+1)
                    '''
                    summary, dl, gl, fl = sess.run([model.summary_op_src, \
                        model.d_loss_src, model.g_loss_src, model.f_loss_src], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl, fl))
                    '''
                
                # train the model for target domain T
                j = step % int(metal_rock_images.shape[0] / self.batch_size)
                trg_images = metal_rock_images[j*self.batch_size:(j+1)*self.batch_size]
                # j = step % int(1748 / self.batch_size)
                # trg_images = img_read_partial(self.metal_rock_dir, j, self.batch_size)
                # trg_images = resize_images(trg_images)
                # trg_images = trg_images / 127.5 - 1
                feed_dict = {model.src_images: src_images, model.trg_images: trg_images}
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)

                if (step+1) % 10 == 0:
                    print ('target - step : ', step+1)
                    '''
                    summary, dl, gl = sess.run([model.summary_op_trg, \
                        model.d_loss_trg, model.g_loss_trg], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl))
                    '''

                if (step+1) % 200 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
                    print ('model/dtn-%d saved' %(step+1))
                
    def eval(self):
        # build model
        model = self.model
        model.build_model()

        # load classical dataset
        classical_images = self.load_classical(self.classical_dir)
        # print (classical_images[0][0].shape)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = classical_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)
                # print (sampled_batch_images.shape)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
                sampled_batch_images = (sampled_batch_images + 1)*127.5
                merged = (merged+1)*127.5
                # print (batch_images[0])
                # scipy.misc.imsave(path, merged)
                # print (sampled_batch_images.shape)
                cv2.imwrite(path, sampled_batch_images[0])
                print ('saved %s' %path)
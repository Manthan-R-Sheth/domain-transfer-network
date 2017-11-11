import tensorflow as tf
from model import DTN
from solver import Solver
import resource

resource.getrlimit(resource.RLIMIT_NOFILE)

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
FLAGS = flags.FLAGS

def main(_):
    
    model = DTN(mode=FLAGS.mode, learning_rate=0.0003)
    solver = Solver(model, batch_size=1, pretrain_iter=20000, train_iter=10000, sample_iter=5, 
                    svhn_dir='classical1', mnist_dir='metal', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
    
    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
        tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
        tf.gfile.MakeDirs(FLAGS.sample_save_path)
    
    if FLAGS.mode == 'pretrain':
        solver.pretrain()
    elif FLAGS.mode == 'train':
        solver.train()
    else:
        solver.eval()
        
if __name__ == '__main__':
    tf.app.run()

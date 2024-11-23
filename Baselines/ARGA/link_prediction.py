from __future__ import division
from __future__ import print_function
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf  
tf.disable_v2_behavior()
import settings
from constructor import get_placeholder, get_model, format_data, get_optimizer, update,format_subgraph_data
from metrics import linkpred_metrics
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Link_pred_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.nb_node_samples = settings['nb_node_samples']
        self.measure = settings['measure']


    def erun(self):
        model_str = self.model
        # formatted data
        feas = format_data(self.data_name)
        feas_sub = format_subgraph_data(self.data_name, self.nb_node_samples,self.measure)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])
        placeholders_sub = get_placeholder(feas_sub['adj'])

        # construct model
        #d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])
        d_real, discriminator, ae_model = get_model(model_str, placeholders_sub, feas_sub['num_features'], feas_sub['num_nodes'], feas_sub['features_nonzero'])

        # Optimizer
        #opt = get_optimizer(model_str, ae_model, discriminator, placeholders_sub, feas_sub['pos_weight'], feas['norm'], d_real, feas['num_nodes'])
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders_sub, feas_sub['pos_weight'], feas_sub['norm'], d_real, feas_sub['num_nodes'])

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        val_roc_score = []

        # Train model
        for epoch in range(self.iteration):
            # Train model and optimize the model
            #emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
            emb, avg_cost = update(ae_model, opt, sess, feas_sub['adj_norm'], feas_sub['adj_label'], feas_sub['features'], placeholders_sub, feas_sub['adj'])


            lm_train = linkpred_metrics(feas['val_edges'], feas['val_edges_false'])
            roc_curr, ap_curr, _ = lm_train.get_roc_score(emb, feas)
            val_roc_score.append(roc_curr)

            
            if (epoch+1) % 25 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost), "val_roc=", "{:.5f}".format(val_roc_score[-1]), "val_ap=", "{:.5f}".format(ap_curr))
                lm_test = linkpred_metrics(feas['test_edges'], feas['test_edges_false'])
                roc_score, ap_score,_ = lm_test.get_roc_score(emb, feas)
                print('Test ROC score: ' + str(roc_score))
                print('Test AP score: ' + str(ap_score))
import settings

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner


dataname = 'pubmed'       # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_vae'          # 'arga_ae' or 'arga_vae'
task = 'link_prediction'         # 'clustering' or 'link_prediction'
nb_node_samples = 2500
measure = 'sn'
sampled_nodes = [500, 1000, 2500,3000,4500,6000]

settings = settings.get_settings(dataname, model, task, nb_node_samples,measure)

if task == 'clustering':
    runner = Clustering_Runner(settings)
else:
    runner = Link_pred_Runner(settings)

runner.erun()


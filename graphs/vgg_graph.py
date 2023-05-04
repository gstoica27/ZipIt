import pdb
import sys
from graphs.base_graph import BIGGraph, NodeType

class VGGGraph(BIGGraph):
    
    def __init__(self, model, architecture):
        super().__init__(model)
        self.architecture = architecture
        
    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        node_insert = NodeType.PREFIX
        graph = []

        graph_idx = 0
        for arch_idx, elem in enumerate(self.architecture):
            if elem == 'M':
                graph.append('features.' + str(graph_idx)) # MaxPool2d
                graph_idx += 1
            else:
                graph.append('features.' + str(graph_idx)) # Conv2d
                graph.append('features.' + str(graph_idx + 1)) # ReLU
                graph.append(node_insert)
                graph_idx += 2
        # pdb.set_trace()
        graph.append('features.' + str(graph_idx)) # AvgPool2d
        graph.append('classifier') # Linear
        graph.append(NodeType.OUTPUT)
        self.add_nodes_from_sequence('', graph, input_node, sep='')

        return self


def vgg11(model):
    architecture = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGGGraph(model, architecture)

def vgg16(model):
    architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGGGraph(model, architecture)

if __name__ == '__main__':
    sys.path.insert(0, '/nethome/gstoica3/research/ModelMerging/')
    import pdb
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from models.vgg import vgg11 as vgg11_model, vgg16 as vgg16_model
    

    from model_merger import ModelMerge
    from matching_functions import match_tensors_identity, match_tensors_zipit
    from copy import deepcopy

    data_x = torch.rand(4, 3, 32, 32)
    data_y = torch.zeros(4)

    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=4)
    
    model = vgg11_model().eval()
    state_dict = model.state_dict()

    print(model)
    model3 = vgg11_model().eval()
    
    graph1 = vgg11(deepcopy(model)).graphify()
    graph2 = vgg11(deepcopy(model)).graphify()

    merge = ModelMerge(graph1, graph2)
    merge.transform(model3, dataloader, transform_fn=match_tensors_identity)

    graph1.draw(nodes=range(len(graph1.G)-20, len(graph1.G)))
    graph1.draw(
        save_path='./graphs/vgg16_graph_auto.png'
    )

    print(model.eval().cuda()(data_x.cuda()))

    print(merge(data_x.cuda()))

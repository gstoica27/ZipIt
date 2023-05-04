from graphs.base_graph import BIGGraph, NodeType


class SinGANGraph(BIGGraph):
    
    def __init__(self, model):
        super().__init__(model)


    def add_basic_block_nodes(self, name_prefix, input_node):
        return self.add_nodes_from_sequence(
            name_prefix=name_prefix,
            list_of_names=['conv', 'batch_norm', 'lrelu', NodeType.POSTFIX],
            input_node=input_node
        )

    # def add_vanilla_nodes(self, name_prefix, input_z, is_last=False, is_first=False):
    #     for block in range(len(self.get_module(name_prefix).features)):
    #         input_z = self.add_basic_block_nodes(f'{name_prefix}.features.{block}', input_z)

    #     input_z = self.add_nodes_from_sequence(
    #         name_prefix=f'{name_prefix}',
    #         list_of_names=['features_to_image.0', 'features_to_image.1'],
    #         input_node=input_z
    #     )
    #     x_inter = self.create_node(layer_name=name_prefix + '.x_intercept', node_type=NodeType.MODULE)
    #     x_z_sum = self.add_nodes_from_sequence(name_prefix, [NodeType.SUM], x_inter)
    #     self.add_directed_edge(input_z, x_z_sum)
    #     if not is_last and not is_first:
    #         output = self.add_nodes_from_sequence(name_prefix, ['sum_intercept', NodeType.POSTFIX], x_z_sum)
    #     else:
    #         output = self.add_nodes_from_sequence(name_prefix, ['sum_intercept'], x_z_sum)
    #     return output
        # return input_z
        
    def add_vanilla_nodes(self, name_prefix, input_z, is_last=False, **kwargs):
        for block in range(len(self.get_module(name_prefix).features)):
            input_z = self.add_basic_block_nodes(f'{name_prefix}.features.{block}', input_z)
            
        list_of_names = [
            'features_to_image.0',
            'features_to_image.1',
            NodeType.PREFIX,
            'intercept1', 
        ]
        output_z = self.add_nodes_from_sequence(name_prefix=f'{name_prefix}', list_of_names=list_of_names,
            input_node=input_z
        )
        
        if not is_last:
            output_p = self.create_node(node_type=NodeType.POSTFIX, special_merge='match_tensors_identity')
            self.add_directed_edge(output_z, output_p)
            return output_p

        return output_z


    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)

        num_scales = len(list(self.get_module('g_model.prev').children()))
        for scale in range(num_scales):
            input_node = self.add_vanilla_nodes(f'g_model.prev.s{scale}', input_node, is_first=scale == 0)
        
        input_node = self.add_vanilla_nodes('g_model.curr', input_node, is_last=True)
        input_node = self.add_nodes_from_sequence('', [NodeType.OUTPUT], input_node, sep='')
        return self


if __name__ == '__main__':
    from model_merger import ModelMerge
    from matching_functions import match_tensors_identity, match_tensors_zipit
    from copy import deepcopy
    from models.singan import Sampler
    s = Sampler()
    s._init_eval(
        # './checkpoints/singan/v4/balloon_models', 
        './checkpoints/singan/balloons/496x372/model',
        '/nethome/gstoica3/research/SinGAN/paired_images/balloon_plane/balloons_496x372.png',
        device='cuda'
    )

    for n, m in s.named_modules():
        print(n)
    
    graph1 = SinGANGraph(s).graphify()
    graph2 = SinGANGraph(deepcopy(s)).graphify()
    
    graph1.draw(save_path='singan_graph_last100.png', nodes=range(150, 260))
    


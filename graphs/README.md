# Graphs
This file contains all instructions for adding new graphs to the repository. At first this may seem daunting. However, we hope that our current code may make the process seemless. Adding a graph requires knowledge of two concepts.

#### Concept 1: Architecture
Each graph is based off a specific kind of model architecture. For instance, a ResNet graph supports resnet architectures, while a VGG graph supports a VGG architecture. Creating a graph requires knowledge of the model architecture you would like to use. 

#### Concept 2: Space changes
Graphs are critical to knowing where to merge in our framework. Specifically because they define unique nodes in charge for dictating the exact location where an alignment between layers of multiple models should be computed. Typically, these positions center around areas of a network where a space *changes*. For instance, the output of any projection operation (e.g., convolution or linear layer), activation, etc... In each of these positions, it is important to demarkate a "space change" with one of two nodes. POSTFIX or PREFIX. A POSTFIX node denotes that the output space of its parent node is different from its parent node's input space. In contrast, a PREFIX node denotes that the input space of its child node is different from the output space of the same child node. When creating your graph, please use these nodes to define where spaces change, and thus where to perform merge/unmerge operations to connect base models together. 

Please see our currently implemented graphs for a demonstration on how to setup your graph. Note that you can run any graph we have implemented by
```
$bash: python -m graphs.resnet_graph
```
Which will create an entire resnet graph architecture, and save a visualization of it to a specified png in the main function. These visualizations can be used to better understand our graphs, and are a very helpful starting point for defining your own. 

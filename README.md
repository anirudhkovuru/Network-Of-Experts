# Network of Errors

##How to run
`python3 train_generalist.py`

Running this will train the generalist part of our model, print the training and testing accuracies and save the model 
to [cifar_net.pth](./Model/cifar_net.pth).
It also gets the Class-To-Specialty mappings and saves the specialties created into [specs.json](./Model/specs.json)

`python3 train_experts.py`

Running this will train the experts in our NofE model using the pretrained [generalist feautures](./Model/cifar_net.pth)
 and [learned specialties](./Model/specs.json), print out the training and testing accuracies as well as save the entire 
 NofE architecture features into 
[nofe.pth](./Model/nofe.pth)

`python3 print_specialties.py`

Running this will print out the specialties in a human-readable format. This is useful to determine if the Generalist 
has properly learned the features required


So the proper order to run the code is to do
```
python3 train_generalist.py
python3 train_experts.py
```
and we will have trained the NofE model and saved the model features and specialty mappings.

## The idea behind the Model

Our project is to implement this paper. 
[Network of Experts for Large-Scale Image Categorization](https://arxiv.org/pdf/1604.06119.pdf)

The method described by the paper is to subdivide the image classification problem into two tasks.

1. Create subgroups of classes which contain classes that share an adequate amount of features with other classes in 
that subgroup. These subgroups of classes are called Specialties.
2. For each specialty, train an 'expert' classifier that can classify images between classes contained in that specialty.

The real life analogy to this model is this; The average layman can recognize effectively most basic visual classes. 
However, discrimination of categories in specific domains requires expert knowledge that can be acquired only through 
dedicated training.
For eg: An average joe can identify that something is a quadruped animal. However, discerning if that animal is a lion, 
panther, leopard requires more expertise knowledge.

In a sense, the visual system of a layman is a very good __*generalist*__ that can accurately
discriminate coarse categories but lacks the specialist eye to differentiate fine
categories that look alike. Becoming an __*expert*__ in any of the aforementioned domains involves time-consuming practical 
training aimed at specializing our visual system to recognize the subtle features that differentiate the given classes.

So our model trains a **Generalist** to discrimate coarse categories and create specialists, and for each specialist we
train an expert that can differentiate fine categories that look alike.

## Method (or How we implemented the model)

Given below is the architecture of the model. The task is to train the main Generalist trunk, arrive at K specialties 
and train K experts for each specialty.

![Network of Experts Model](http://vlg.cs.dartmouth.edu/projects/nofe/approach.jpg)

- Top part of the model shows the generalist, which outputs the k branches and acts as the trunk
- Bottom part shows the whole NofE model with the convolution layers of the generalist acting as a trunk branching into 
K branches with each branch attaching to a CNN which trains on it to become an expert.

The generalist is implemented in the form of a convolutional neural network (CNN) with a final softmax layer over *K* 
specialties, where *K* << *C*, with *C* denoting the original number of classes. After this first training stage, the fully
connected layers are discarded and *K* distinct branches are attached to the last 3 convolutional layer of the generalist,
i.e., one branch per specialty. Each branch is associated to a specialty and is devoted to recognize the classes within the
specialty. 

This gives rise to the NofE architecture, a unified tree-structured network. Finally, all layers of the resulting 
model are finetuned with respect to the original C categories by means of a global softmax layer that calibrates the outputs 
of the individual experts over the C categories.

#### Generalist

To train the Generalist(that is, optimize the generalist parameters Ө<sup>G</sup>), we need the mapping ℓ from class to 
specialties, which are unknown in the beginning. We have a
chicken and egg problem here.


So, we initialize the mapping ℓ by first randomly partitioning the *C* classes into *K* subsets, such that each subset 
has equal size (*C*/*K* size).
 The Generalist is then trained using an alternation scheme that follows so:
1. Optimize Ө<sup>G</sup> by training over the entire dataset while keeping specialty mapping ℓ fixed.
2. Evaluate the generalist defined by Ө<sup>G</sup> over a random subset S of our training data. For this subset we build 
the confusion matrix M&isin;R<sup>*C*&times;*K*</sup> where *M<sub>ij</sub>* is the fraction of examples of class label 
*i* that are classified into
specialty *j* by the current generalist. We create new specialty mappings ℓ by assigning a randomly chosen class to that 
specialty which has the highest value *M<sub>ij</sub>* among the specialties that have not yet reached maximum size *C/K*
, for all classes in C. With our new mappings ℓ, we go back to step 1 and the cycle continues.

Given below is the Generalist architecture:
![Generalist Architecture](https://www.researchgate.net/profile/Mohammadharis_Baig/publication/301835547/figure/tbl2/AS:667112569847809@1536063439093/AlexNet-C100-trained-on-CIFAR100.png)
The generalist contains 3 convolutional layers and 1 fully connected layer. It is a modification of AlexNet architecture
which has 5 CONV layers and 3 fully connected layers. The model uses a smaller version of Alexnet in order to deal with
the smaller dataset used(CIFAR-100 is much smaller than ImageNet on which AlexNet is trained, so the generalist doesnt
require as many features.

#### Experts

The Experts are much easier to train in comparison. Given the generalist Ө<sup>G</sup> and the class-to-specialty mapping
ℓ produced by the first stage of training, we perform joint learning of the *K* experts in order to
obtain a global multi-class classification model over the original classes *C*. This is achieved by defining a tree-structured
network consisting of a single trunk feeding K branches, one branch for each specialty. The trunk is initialized with the 
convolutional layers of the generalist, as they have been optimized to yield accurate specialty classification. 

Since each branch is responsible for discriminating only the classes associated to its specialty, the number of output units of
the last FC layer is *C/K*. We apply a global softmax over all the *K* branches (which gives K&times;C/K = C labels). We do joint
training on all branches at the same time.

Each branch contains one convolutional layers followed by one fully-connected layer, as seen in the above image displaying
the Generalist and NofE architecture.

## Results

####Generalist

After training the generalist, we tested its accuracy, and achieved results = 50.11% (basically 1
out of 2 classes were put in the correct specialty). The specialties we got (with K=10) are:-
- Speciality 0 : ['whale', 'flatfish', 'otter', 'seal', 'turtle', 'crocodile', 'shark', 'worm', 'mountain', 'dolphin']
- Speciality 1 : ['lobster', 'bee', 'butterfly', 'crab', 'rose', 'aquarium_fish', 'tulip', 'poppy', 'orchid', 'sweet_pepper']
- Speciality 2 : ['bed', 'television', 'wardrobe', 'cup', 'table', 'couch', 'lamp', 'clock', 'bowl', 'ray']
- Speciality 3 : ['mushroom', 'palm_tree', 'willow_tree', 'forest', 'dinosaur', 'pine_tree', 'maple_tree', 'shrew', 'oak_tree', 'porcupine']
- Speciality 4 : ['road', 'sunflower', 'lawn_mower', 'plain', 'skunk', 'leopard', 'orange', 'cockroach', 'trout', 'caterpillar']
- Speciality 5 : ['tractor', 'pickup_truck', 'streetcar', 'train', 'house', 'castle', 'bridge', 'tank', 'bus', 'raccoon']
- Speciality 6 : ['possum', 'cattle', 'wolf', 'squirrel', 'tiger', 'beaver', 'lion', 'bear', 'rabbit', 'mouse']
- Speciality 7 : ['rocket', 'skyscraper', 'sea', 'chair', 'cloud', 'can', 'bottle', 'telephone', 'pear', 'apple']
- Speciality 8 : ['bicycle', 'spider', 'motorcycle', 'plate', 'keyboard', 'snake', 'kangaroo', 'camel', 'snail', 'lizard']
- Speciality 9 : ['baby', 'chimpanzee', 'hamster', 'girl', 'woman', 'fox', 'man', 'boy', 'elephant', 'beetle']

We talk about the interesting observations and patterns found in our Project Evaluation presentation

#### Whole NofE model

After training the experts, and combining them with our generalist to build our NofE
architecture, we achieved accuracy of 48.26% in classification. This is similar to the accuracy
presented in the paper(56.1%) with K=10.


## Libraries Used

- torch
- torchvision
- Numpy
- Pickle
- JSON
- random
- os
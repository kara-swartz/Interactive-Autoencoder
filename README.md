# A Visual Interactive Latent Space Exploration for Autoencoders (AE) and Variational Autoencoders (VAE)

Analyzing high-dimensional data can be challenging due to the curse of dimensionality. As the number of dimensions increases, data becomes highly sparse. Due to a large number of properties, such data is difficult to understand and interpret, the relationship between the properties may become unclear. Dimensionality Reduction (DR) are machine learning techniques used to reduce the number of dimensions. Autoencoder (AE) is a type of neural network that is widely used for DR. The AE encodes input data into a low-dimensional representation (latent space) and tries to reconstruct the compressed data. AE, like other neural networks models, are traditionally considered as a "black box", so it is difficult to interpret what the model has learned and what relationship between input data and the resulting latent space representation.

Here we present a visualization system for visual interactive latent space exploration that allows data analysts to dynamically specify a region of interest in the latent space and observe the impact of these changes. Our motivation is to get a deeper insight into the learning process of the AE models, for this, we visualize the distribution of the latent space. The system allows the data analysts to define an area of interest in the latent representation by the class selection, random point selection, and linking & brushing. Interactive visualization helps to observe the impact of different point selection scenarios on the resulting latent space.

In addition to the AE, the system visualizes the latent space obtained by the Variational Autoencoder (VAE), a generative neural network that learns the probabilistic distribution of input data. Thus, we present two neural network models that have been trained on different data sets, such as iris, wine quality, air quality, and MNIST.

## Directory Structure

```
interactive-autoencoder
|-- app 
|   |-- models (pytorch models)
|       |--ae_mnist.py
|       |--ae_wine.py
|       |--vae_mnist.py
|       |--vae_wine.py
|   |--templates
|       |-- layout.html
|       |-- index.hrml
|   |-- app.py
```

## Run Flask Ap

Run flask in debugging mode:
```
export FLASK_ENV=development
flask run
```

## System Design

The central element of the visual interactive system is the web application server, which is developed using the Flask web framework. The PyTorch framework is used to develop neural network models. We used the D3.js JavaScript library to implement interactive visualization.

![Image of System Design](https://github.com/kara-swartz/Interactive-Autoencoder/blob/main/app/static/images/system_design.png)

After the user selects the neural network model and data set, the Flask server calls the script to start the training of this model. AE or VAE model converts the input data into a two-dimensional representation. The latent space is displayed on the first plot. Further, the user selects the area of interest, and then the Flask server forwards the selected points to the neural network input for additional training.

The neural network model provides training on the selected points and calculates the values of quality measures. Lastly, the server submits the new latent representation in the visualization. Javascript function separates the selected points from the remaining points and then renders them in the second and third scatter plots.

## Core literature and resources

### Theoretical framework
* Paper: Hinton, Salakhutdinov - Reducing the Dimensionality of Data with Neural Networks
* Video: Hinton - [From PCA to autoencoders](https://www.youtube.com/watch?v=hbU7nbVDzGE)
* Paper: Cavallo, Çağatay - A Visual Interaction Framework for Dimensionality Reduction Based Data Exploration
* T. Spinner, J. Körner, J. Görtler, and O. Deussen. - [Towards an interpretable latent space](https://thilospinner.com/towards-an-interpretable-latent-space)

### Technical resources
* flask + pytorch - [The brilliant beginner’s guide to model deployment](https://heartbeat.fritz.ai/brilliant-beginners-guide-to-model-deployment-133e158f6717)
* flask + tensorflow - [Serving a model with Flask](https://guillaumegenthial.github.io/serving.html)
* mozilla docs - [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
* d3-graph-gallery [Brushing in d3.js](https://www.d3-graph-gallery.com/graph/interactivity_brush.html)
* observablehq, Bostock - [Posting with Fetch](https://observablehq.com/@mbostock/posting-with-fetch)

# A Visual Interactive Latent Space Exploration for Autoencoders (AE) and Variational Autoencoders (VAE)

Analyzing high-dimensional data can be challenging due to the curse of dimensionality. As the number of dimensions increases, data becomes highly sparse. Due to a large number of properties, such data is difficult to understand and interpret, the relationship between the properties may become unclear. Dimensionality Reduction (DR) are machine learning techniques used to reduce the number of dimensions. Autoencoder (AE) is a type of neural network that is widely used for DR. The AE encodes input data into a low-dimensional representation (latent space) and tries to reconstruct the compressed data. AE, like other neural networks models, are traditionally considered as a "black box", so it is difficult to interpret what the model has learned and what relationship between input data and the resulting latent space representation.

Here we present a visualization system for visual interactive latent space exploration that allows data analysts to dynamically specify a region of interest in the latent space and observe the impact of these changes. Our motivation is to get a deeper insight into the learning process of the AE models, for this, we visualize the distribution of the latent space. The system allows the data analysts to define an area of interest in the latent representation by the class selection, random point selection, and linking & brushing. Interactive visualization helps to observe the impact of different point selection scenarios on the resulting latent space.

In addition to the AE, the system visualizes the latent space obtained by the Variational Autoencoder (VAE), a generative neural network that learns the probabilistic distribution of input data. Thus, we present two neural network models that have been trained on different data sets, such as iris, wine quality, air quality, and MNIST.


## Run Flask Ap

Run flask in debugging mode:
```
export FLASK_ENV=development
flask run
```

## Directory Structure

```
interactive-autoencoder
|   |-- app
|       |-- app.py
|       |-- model.py
|       |-- model_2.py
|   |-- templates
|       |-- index.hrml
|       |-- iris.csv
```

## Literature and resources

### Milestone 1
* Paper: Cavallo, Çağatay - A Visual Interaction Framework for Dimensionality Reduction Based Data Exploration
* Paper: Hinton, Salakhutdinov - Reducing the Dimensionality of Data with Neural Networks
* Video: Hinton - [From PCA to autoencoders](https://www.youtube.com/watch?v=hbU7nbVDzGE)
* flask + pytorch - [The brilliant beginner’s guide to model deployment](https://heartbeat.fritz.ai/brilliant-beginners-guide-to-model-deployment-133e158f6717)
* flask + tensorflow - [Serving a model with Flask
](https://guillaumegenthial.github.io/serving.html)

### Milestone 2
* mozilla docs - [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
* d3-graph-gallery [Brushing in d3.js](https://www.d3-graph-gallery.com/graph/interactivity_brush.html)
* observablehq, Bostock - [Posting with Fetch](https://observablehq.com/@mbostock/posting-with-fetch)
* stackoverflow - [Send POST request in d3 with d3-fetch](https://stackoverflow.com/questions/51650427/send-post-request-in-d3-with-d3-fetch)

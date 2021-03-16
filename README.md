# Interactive Autoencoder

Autoencoder that allows for user interaction, especially optimizing areas of interest.

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
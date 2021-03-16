from flask import Flask, request, render_template
from models.vae_mnist import LatentSpace as vae_mnist
from models.vae_wine import LatentSpace as vae_wine
from models.ae_mnist import LatentSpace as ae_mnist
from models.ae_wine import LatentSpace as ae_wine

app = Flask(__name__)

instance = None

@app.route('/')
def home():
    return render_template("layout.html")

@app.route('/model/<model>', methods=['GET', 'POST'])
def index(model):
    if request.method == 'GET':
        global instance
        #start instance
        if model == 'vae_wine':
            instance = vae_wine()
        elif model == 'vae_mnist':
            instance = vae_mnist()
        elif model == 'ae_wine':
            instance = ae_wine()
        elif model == 'ae_mnist':
            instance = ae_mnist()
        output = instance.get_latent()
        output_data = output[0]
        qm_mse_all_points_pr1 = output[1]
        output_data = output_data.to_json(orient='records')
        data = {'output_data' : output_data, 'url_title' : model}
        return render_template("index.html", data = data, qm_mse_all_points_pr1 = qm_mse_all_points_pr1
        )  
    if request.method == 'POST':
        selected = request.get_json()
        #convert array of strings to array of numbers
        selected = list(map(int, selected))
        output = instance.projection(selected)
        projection = output[0]
        projection['qm_mse_selected_pr1'] = output[1]
        projection['qm_mse_all_points_pr2'] = output[2]
        projection['qm_mse_selected_pr2'] = output[3]
        return projection.to_json(orient='records')
        
if __name__ == '__main__':
    app.run(debug=True)
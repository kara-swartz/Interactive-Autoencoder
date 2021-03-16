""" 
    second prototype
    two instance model
"""
from flask import Flask, request, render_template, jsonify
from models.ae_model import AutoEncoder
from models.vae_model import VAE
from models.quality_measures import qm_mae_dist, qm_mse_dist, qm_corr_dist
from sklearn import preprocessing
from sklearn.datasets import load_iris, fetch_openml

import pandas as pd
import numpy as np
import json

app = Flask(__name__)
    
######### loading data #########
# white wine quality data set
data_white = pd.read_csv("../app/static/data/winequality-white.csv", sep =";")
data_red = pd.read_csv("../app/static/data/winequality-red.csv", sep =";")
df = pd.concat([data_red, data_white])
df = df.drop(columns=['quality'])

# air quality data set
""" df = pd.read_csv("static/data/airquality.csv")
df = df.drop(columns=['Date','Time','NMHC(GT)']) """

# data noramlization
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df.astype(float))
df = pd.DataFrame(x_scaled)

# convert to numpy array
x_data = df.to_numpy()

# iris data
""" iris = load_iris()
x_data=iris.data """
########## ########## ##########

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        #start instance
        global inst_1
        inst_1 = AutoEncoder()#VAE()
        inst_1.get_data(x_data)        
        output = inst_1.train()
        encoded = output[0]
        decoded = output[1]
        global v_decoded
        v_decoded = decoded            
        output_data = pd.DataFrame(encoded,columns=['x', 'y'])
        output_data['trained'] = 'no'      
        output_data = output_data.to_json(orient='records')
        data = {'output_data' : output_data}
            
            #Quality measures. Projection 1
        v_qm_mae_dist_all_1 = round(qm_mae_dist(x_data,decoded), 4)
        v_qm_mse_dist_all_1 = round(qm_mse_dist(x_data,decoded), 4)
        print(f"""
        All points. Projection 1:
        Mean absolute error of distances is {v_qm_mae_dist_all_1}
        Mean squared error of distances is {v_qm_mse_dist_all_1}
        """)
        return render_template("index.html", data = data, v_qm_mse_dist_all_1 = v_qm_mse_dist_all_1)
    if request.method == 'POST':
        selected = json.loads(request.data)
        #convert array of strings to array of numbers
        selected = list(map(int, selected))                      
        decoded = np.asarray(v_decoded)
            
            #Quality measures. Projection 1
        v_qm_mae_dist_selected_1 = round(qm_mae_dist(x_data[selected],decoded[selected]), 4)
        v_qm_mse_dist_selected_1 = round(qm_mse_dist(x_data[selected],decoded[selected]), 4)
        print(f"""
        Selected points. Projection 1:
        Mean absolute error of distances is {v_qm_mae_dist_selected_1}
        Mean squared error of distances is {v_qm_mse_dist_selected_1}
        """)
        
            # projection
        projection_data = inst_1.projection(x_data)
        projection_encoded = projection_data[0]
        projection_decoded = projection_data[1]
        #convert to pandas in order to add column's names
        projection = pd.DataFrame(projection_encoded,columns=['x', 'y'])            
        #added columns name here because first data set is also without headers
        projection['trained'] = 'no'                                         
        #to seperate colors for trained and projected points
        projection.loc[selected, 'trained'] = 'yes'                       
        
            #Quality measures. All points. Projection 2:
        v_qm_mae_dist_all_2 = round(qm_mae_dist(x_data,projection_decoded), 4)
        v_qm_mse_dist_all_2 = round(qm_mse_dist(x_data,projection_decoded), 4)
        print(f"""
        All points. Projection 2:
        Mean absolute error of distances is {v_qm_mae_dist_all_2}
        Mean squared error of distances is {v_qm_mse_dist_all_2}
        """)
            
            #Quality measures. Selected points. Projection 2:
        v_qm_mae_dist_selected_2 = round(qm_mae_dist(x_data[selected],projection_decoded[selected]), 4)
        v_qm_mse_dist_selected_2 = round(qm_mse_dist(x_data[selected],projection_decoded[selected]), 4)
        print(f"""
        Selected points. Projection 2:
        Mean absolute error of distances is {v_qm_mae_dist_selected_2}
        Mean squared error of distances is {v_qm_mse_dist_selected_2}
        """)
        projection['v_qm_mse_dist_selected_1'] = v_qm_mse_dist_selected_1
        projection['v_qm_mse_dist_all_2'] = v_qm_mse_dist_all_2
        projection['v_qm_mse_dist_selected_2'] = v_qm_mse_dist_selected_2
        return projection.to_json(orient='records')
        
if __name__ == '__main__':
 app.run(debug=True)
import warnings
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template

from test_model import perform_prediction, data_preprocessing

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
app = Flask(__name__)
cors = CORS(app)


@app.route('/home',methods = ['POST','GET'])
def home():
    return render_template('home.html')

@app.route('/check',methods = ['POST','GET'])
def check():
    crime_rate = float(request.args.get('crimeRate'))
    print (crime_rate,"dadddd")
    return crime_rate

@app.route('/predict', methods=['POST' ,'get'])
def load_inference():
    # crime_rate = float(request.args.get('crimeRate'))
    # residential_land_zone_area = float(request.args.get('residentialLandZoneArea'))
    # non_retail_business_area = float(request.args.get('nonRetailBusinessArea'))
    # river = request.args.get('river')
    # nitric_oxides_level = float(request.args.get('nitricOxidesLevel'))
    # average_room_count = float(request.args.get('averageRoomCount'))
    # house_age = float(request.args.get('houseAge'))
    # industrial_location_distance = float(request.args.get('industrialLocationDistance'))
    # radial_highways_count = float(request.args.get('radialHighwaysCount'))
    # property_tax_rate = int(request.args.get('propertyTaxRate'))
    # pupil_teacher_ratio = float(request.args.get('pupilTeacherRatio'))
    # migrant_population_ratio = float(request.args.get('migrantPopulationRatio'))
    # rural_population_ratio = float(request.args.get('ruralPopulationRatio'))

    # crime_rate = 0.00632
    # residential_land_zone_area = 18
    # non_retail_business_area = 2.309999943
    # river = 0
    # nitric_oxides_level = 0.537999988
    # average_room_count = 6.574999809
    # house_age = 1.19999695
    # industrial_location_distance = 4.090000153
    # radial_highways_count = 1
    # property_tax_rate = 296
    # pupil_teacher_ratio = 15.30000019
    # migrant_population_ratio = 396.8999939	
    # rural_population_ratio = 4.980000019
    # model = tensorflow.keras.models.load_model('')
    # path = "./hpp.pickle"
    # model = pickle.load(open(path,'rb'))
    # model.predict([3,2,150])
# 0.00632	18	2.309999943	0	0.537999988	6.574999809	65.19999695	4.090000153	1	296	15.30000019	396.8999939	4.980000019	24


# use this in postman
# crim:0.00632
# zn:18
# indus:2.30999
# chas:0
# nox:0.537999988
# rm:6.574999809
# age:65.19999695
# dis:4.090000153
# rad:1
# tax:296
# ptratio:15.30000019
# b:396.8999939
# lstat:4.980000019

    if request.method == 'POST':
        a = request.form.get('crim')
        b = request.form.get('zn')
        c = request.form.get('indus')
        d = request.form.get('chas')
        e = request.form.get('nox')
        f = request.form.get('rm')
        g = request.form.get('age')
        h = request.form.get('dis')
        i = request.form.get('rad')
        j = request.form.get('tax')
        k = request.form.get('ptratio')
        l = request.form.get('b')
        m = request.form.get('lstat')

    print('Data=', a,b,c,d,e,f,g,h,i,j,k,l,m)

    column_value = [a,b,c,d,e,f,g,h,i,j,k,l,m]
    column_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PT', 'B', 'LSTAT']

    data_frame = pd.DataFrame([column_value], columns=column_name)
    s
    data_preprocessing(data_frame, "CHAS")
    prediction = perform_prediction(data_frame)
    print('prediction>>>>', prediction, type(prediction))
    house_price = (prediction.tolist())[0]
    response_dict = {'house_price': house_price}
    print('response_dict>>>>', response_dict, type(response_dict))
    response = jsonify(response_dict)
    print('response>>>>', response, type(response))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.debug = True
    app.run()

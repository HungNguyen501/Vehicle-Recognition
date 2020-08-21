import numpy as np
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import sys
import pickle
from flask import Flask, render_template, request, send_from_directory

# initialize the local binary patterns descriptor along with
# the data and label lists
modelFileName = 'Model.sav'
model = LinearSVC(C=100.0, random_state=42, max_iter=10000)
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []    

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", default='./images/test2/training',
	help="path to the training images")
ap.add_argument("-e", "--testing", default='./images/test2/testing',
	help="path to the testing images")

args = vars(ap.parse_args('')) # get default arguments


# Check Model File exists or not
if not (os.path.exists('./' + modelFileName) ) :
    #If model file not exists
    print('Training model...')
    
    # loop over the training images
    for imagePath in paths.list_images(args["training"]):
        # load the image, convert it to grayscale, and describe it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.lbp(gray)   
        
        # extract the label from the image path, then update the
        # label and data lists      
        labels.append(imagePath.split(os.path.sep)[-2])   
        
        #data.append(np.hstack([hist1, hist2]))
        data.append(hist)
    
    # Train a Linear SVM on the data
    model.fit(data, labels)    
    # Save model
    pickle.dump(model, open(modelFileName, 'wb'))
    
    # Show accuracy of the model
    result = model.score(data, labels)
    print('Accuracy of the model: ', result)
    
else:
    # Load the model from disk
    model = pickle.load(open(modelFileName, 'rb'))

def show_output(model, imagePath):    
    querryImage = cv2.imread(imagePath)
    gray = cv2.cvtColor(querryImage, cv2.COLOR_BGR2GRAY)
    hist = desc.lbp(gray)
   
    prediction = model.predict(hist.reshape(1, -1))

	# display the image and the prediction
    cv2.putText(querryImage, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 100), 3)
    # Save output image to file
    outputImagePath="./output/sc.jpg"
    cv2.imwrite(outputImagePath, querryImage)
    
    return prediction[0]

app = Flask(__name__, template_folder='.')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template("upload.html") 

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'output/')
    #print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for upload in request.files.getlist("input_image"): 
        filename= upload.filename
        image_path = "./images/test2/testing/" + filename
        #print("\n___imagePath: " + image_path)
        destination = "/".join([target, filename])
        #print("\n___destination: ")
        #print(destination)
        #upload.save(destination)
        
        #processing detection
        show_output(model, image_path)

    #return send_from_directory("output", filename, as_attachment=True)
    return render_template("complete.html", image_name="sc.jpg")
       
@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("output", filename)   

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__=="__main__":
    app.run(port=4555, debug=True)


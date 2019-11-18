# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F "image=@jemma.png" "http://localhost:5000/predict"
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages

import keras
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io
from keras.models import load_model
import pickle
import os
import imutils

import recognizer.utils.color_correlation as color_correlation
import cv2
import recognizer.utils.dewapper as dewapper
from recognizer.utils.signature_extractor import extract_signature
import recognizer.utils.unsharpen as unsharpen
from PIL import Image
import json
import scipy.misc
import keras.backend.tensorflow_backend as tb
from flask import jsonify

# import flask_cors  as corsnn

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None
lb = None


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a


def prepare_image(image, target):
    # if the image mode is not RGB, convert it

    image = cv2.resize(image, (32, 32))
    # if image.mode != "RGB":
    # image = image.convert("RGB")

    # resize the input image and preprocess it
    # image = image.resize(target)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # this is new
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


def classify_process():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    print("* Loading model...")
    model = load_model("content_model/signature.h5")
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves

        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], IMAGE_DTYPE,
                                        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))

            # check to see if the batch list is None
            if batch is None:
                print("loading image")
                batch = image

            # otherwise, stack the data
            else:
                print("Stacking data ..")
                batch = np.vstack([batch, image])
                cv2.imwrite("bon.jpg", batch)
                cv2.imwrite("soir.jpg", image)
            # update the list of image IDs
            imageIDs.append(q["id"])

        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            # classify the batch
            print("* Batch size: {}".format(batch.shape))
            preds = model.predict(batch)[0]
            print(preds)
            results = imagenet_utils.decode_predictions(preds)

            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(imageIDs, results):
                # initialize the list of output predictions
                output = []

                # loop over the results and add them to the list of
                # output predictions
                for (imagenetID, label, prob) in resultSet:
                    r = {"label": label, "probability": float(prob)}
                    output.append(r)

                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))

            # remove the set of images from our queue
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        # sleep for a small amount
        time.sleep(SERVER_SLEEP)


def initialize_model():
    print("* Loading model...")
    global model
    global lb
    model = load_model("content_model/signature.h5")
    lb = pickle.loads(open("content_model/lb.pickle", "rb").read())
    print("* Model & Labels loaded")


@app.route("/predict", methods=["POST"])
def predict():
    print("RECEIVING STUFFFFFFFF")
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print(flask.request.data)

        data2 = flask.request.data.decode('utf-8')

        bason = json.loads(data2)
        print(bason['image'])
        if flask.request.data:
            # read the image in PIL format and prepare it for

            image = Image.open(r"C:/xampp/htdocs/vision/uploads/{0}".format(bason["image"]))
            # image = flask.request.files["image"]

            # image = Image.open(io.BytesIO(image))

            # prepare some unique ID
            k = str(uuid.uuid4())
            os.mkdir("outputs/{0}".format(k), 0o755)
            os.mkdir("outputs/{0}/extracted".format(k), 0o755)
            image.save(os.path.join("outputs/{0}".format(k), "raw.jpg"))

            # 1 - BEGIN Before ANY CLASSIFICATION -- INPUT : IMAGE

            image = cv2.imread("outputs/{0}/raw.jpg".format(k))

            source_image = image
            img = 0
            try:
                # read the source input image and call the dewarp_book function
                # to perform cropping with the margin and book dewarping
                img = dewapper.dewarp_book(source_image)
                cv2.imwrite("outputs/{0}/extracted/step 1 - page_dewarped.jpg".format(k), img)
                print("- step1 (cropping with the argins + book dewarpping): OK")
            except Exception as e:
                print("type error: " + str(e))
                print("ERROR IN CROPPING & BOOK DEWARPING! PLEASE CHECK LIGTHNING,"
                      " SHADOW, ZOOM LEVEL AND ETC. OF YOUR INPUT BOOK IMAGE!")
            try:

                # call the unsharpen_mask method to perform signature extraction
                img, extracted_images = extract_signature(cv2.cvtColor(img,
                                                                       cv2.COLOR_BGR2GRAY), identifier=k)
                cv2.imwrite("outputs/{0}/extracted/step 2 - signature_extracted.jpg".format(k), img)
                print("- step 2 : Extracted {0} signatures".format(len(extracted_images)))
                print("- step 2 (signature extractor): OK")
            except Exception as e:
                print("type error: " + str(e))
                print("ERROR IN SIGNATURE EXTRACTION! PLEASE CHECK LIGTHNING, SHADOW,"
                      " ZOOM LEVEL AND ETC. OF YOUR INPUT BOOK IMAGE!")
            try:
                # call the unsharpen_mask method to perform unsharpening mask
                unsharpen.unsharpen_mask(img)
                cv2.imwrite("outputs/{0}/extracted/step 3 - unsharpen_mask.jpg".format(k), img)
                print("- step3 (unsharpening mask): OK")
            except Exception as e:
                print("type error: " + str(e))
                print("ERROR IN BOOK UNSHARPING MASK! PLEASE CHECK LIGTHNING, SHADOW,"
                      " ZOOM LEVEL AND ETC. OF YOUR INPUT BOOK IMAGE!")
            try:
                # call the funcBrightContrast method to perform color correction
                img = color_correlation.funcBrightContrast(img)
                cv2.imwrite("outputs/{0}/extracted/step 4 - color_correlated.jpg".format(k), img)
                print("- step4 (color correlation): OK")
            except Exception as e:
                print("type error: " + str(e))
                print("ERROR IN BOOK COLOR CORRELATION! PLEASE CHECK LIGTHNING, SHADOW,"
                      " ZOOM LEVEL AND ETC. OF YOUR INPUT BOOK IMAGE!")

            # 2 - END Before Any Classification --> OUTPUT : Resized image to classify
            # classification

            # run classification on all extracted image

            # prepare results holder
            result_label = ""
            result_array_holder = []
            best_result = 0
            best_label = ""
            best_image = None

            for image in extracted_images:
                # convert back to RGB
                orig = image
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = cv2.resize(image, (32, 32))
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)

                # classify the input image
                print("[INFO] Predicting image...")
                proba = model.predict(image)[0]
                idx = np.argmax(proba)
                label = lb.classes_[idx]

                # build the label and draw the label on the image
                label = "{}: {:.2f}%".format(label, proba[idx] * 100)
                result_array_holder.append(label)
                if best_result < proba[idx] * 100:
                    best_result = proba[idx] * 100
                    best_label = label
                    best_image = orig

            # store best image to XAMPP server
            os.mkdir("C:/xampp/htdocs/signature_repo/{0}".format(k))

            # im = Image.fromarray(best_image.astype('uint8'))
            best_image = cv2.resize(best_image, (96, 96), interpolation=cv2.INTER_AREA)

            cv2.imwrite("C:/xampp/htdocs/signature_repo/{0}/{1}".format(k, "predicted.png"), best_image)

            # indicate that the request was a success
            data["success"] = True
            data["signature_count"] = len(extracted_images)
            data["result"] = best_label
            data["debug"] = json.dumps(result_array_holder)
            data["linktoimage"] = "http://localhost/signature_repo/{0}/{1}".format(k, "predicted.png")
            print(data)
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    print(keras.__version__)
    print("* Starting model service...")
    # classify_process()
    initialize_model()
    """""
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()
    """
    # start the web server
    print("* Starting web service...")
    app.run()

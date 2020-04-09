# MNIST-clasification-
Handwritten sigits are classified and the result is deployed in localhost.
Make a directory named flask_deploy in your current working area.
Create few more directories to save our CNN model and to store HTML, javascript, CSS and Flask files.
flask_deploy
    |---- model
    |       |---- model.h5
    |       |---- model.json
    |
    |---- static
    |       |---- index.js
    |       |---- style.css
    |
    |---- templates
    |       |---- index.html
    |
    |---- keras_flask.py
    
The above 3 files form the front-end of our web app with which the user will interact. 
You will draw a figure with the help of your mouse inside a box which will be converted into an image of size 28x28 and will be passed to our saved model. 
The code to handle this exchange will be taken care by Flask. 
Flask is a micro-framework used to develop websites quickly and it is written in Python.
Once the image is sent in from the web page it runs it though the trained model and gets the prediction and passes it back to the web page to display.
After you are done with this you can go inside the flask_deploy directory and simply run — python keras_flask.py or ipython keras_flask.py
This will launch the flask application and will open a tab in your default browser. 
If it doesn’t, try opening http://localhost:5000 in your browser. 
If you get any errors saying “module not found” just run conda install {module name} in your environment.

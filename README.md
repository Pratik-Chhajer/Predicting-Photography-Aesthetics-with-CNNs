# Predicting Photography Aesthetics with CNNs

### Installation

First, clone the repository

Next, install the required python3 packages: 
```bash
pip3 install -r requirements.txt
```

Now to classify any new images put all your image inside folder named **test_images**
```bash
python3 scripts/main.py
```
It will give output like this:-

```bash
image1_nmae 	 3
image2_name 	 2
image3_name 	 2
image4_name 	 5
```



To evaluate our CNN models run:-
```bash
python3 scripts/evaluate.py
```
This will read all three models saved in model named folder and output something like this:-
```bash
Reading Model 1
Evaluating Mdel 1
800/800 [==============================] - 6s 8ms/step
Test loss for Model 1: 1.76433014154
Test accuracy for Model 1: 0.35
############################################
Reading Model 2
Evaluating Model 2
800/800 [==============================] - 8s 10ms/step
Test loss for Model 2: 1.43449003458
Test accuracy for Model 2: 0.36375
############################################
Reading Model 3
Evaluating Mdel 3
800/800 [==============================] - 4s 5ms/step
Test loss for Model 3: 1.43597866058
Test accuracy for Model 3: 0.41375
############################################ 
```

To build all three models again run(Already done, not required as this will take more than 8 hours):-
```bash
python3 scripts/model1.py
python3 scripts/model2.py
python3 scripts/model3.py
```

This will generate our three models named
```bash
model1.h5
model2.h5
model3.h5
```




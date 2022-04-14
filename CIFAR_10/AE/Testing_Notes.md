### Note about naming convention: 
The names of the files show which class was normal and which was abnormal
For example: AE_CIFAR10_Airplane_Ship.ipynb tests Airplanes as the normal class and Ships as the abnormal class

### AE_CIFAR10_Cat_Ship.ipynb
* model: 
    ```def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(200, activation="selu"),
          layers.Dense(100, activation="selu"),
          layers.Dense(50, activation="selu")])

        self.decoder = tf.keras.Sequential([
          layers.Dense(100, activation="selu"),
          layers.Dense(200, activation="selu"),
          layers.Dense(32*32*3, activation="sigmoid"), # include RGB
          layers.Reshape((32, 32, 3))]) # 3 dimensional because RGB```
* epochs: 200
* batch_size: 512
* Confusion Matrix: 
   ```prediction: F      T 
               4   1996
   label: F   [[0   1000]    1000
          T    [4   996]]   1000
  Accuracy = 0.498
  Precision = 0.49899799599198397
  Recall = 0.996

### AE_CIFAR10_Bird_Frog.ipynb
* model: 
    ```  def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(1024, activation="selu"),
          layers.Dense(512, activation="selu"),
          layers.Dense(256, activation="selu")])

        self.decoder = tf.keras.Sequential([
          layers.Dense(512, activation="selu"),
          layers.Dense(1024, activation="selu"),
          layers.Dense(32*32*3, activation="sigmoid"), # include RGB
          layers.Reshape((32, 32, 3))]) # 3 dimensional because RGB
* epochs: 200
* batch_size: 256
*  Confusion Matrix
    ```prediction: F      T 
                 462   1538
     label: F   [[248   752]    1000
            T    [214   786]]   1000
    Accuracy = 0.517
    Precision = 0.5110533159947984
    Recall = 0.786 

### AE_CIFAR10_Airplane_Ship.ipynb
* model: 
    ```  def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(1024, activation="selu"),
          layers.Dense(512, activation="selu"),
          layers.Dense(256, activation="selu")])

        self.decoder = tf.keras.Sequential([
          layers.Dense(512, activation="selu"),
          layers.Dense(1024, activation="selu"),
          layers.Dense(32*32*3, activation="sigmoid"), # include RGB
          layers.Reshape((32, 32, 3))]) # 3 dimensional because RGB
* epochs: 200
* batch_size: 256
* Confusion Matrix
   ```prediction: F      T 
               531   1469
   label: F   [[324   676]    1000
          T    [207   793]]   1000
  Accuracy = 0.5585
  Precision = 0.5398230088495575
  Recall = 0.793 

### AE_CIFAR10_Airplane_Cat.ipynb
* model:
    ```def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(1024, activation="selu"),
          layers.Dense(512, activation="selu"),
          layers.Dense(256, activation="selu")])
        self.decoder = tf.keras.Sequential([
          layers.Dense(512, activation="selu"),
          layers.Dense(1024, activation="selu"),
          layers.Dense(32*32*3, activation="sigmoid"), # include RGB
          layers.Reshape((32, 32, 3))]) # 3 dimensional because RGB
* epochs: 200
* batch_size: 256
* Confusion Matrix: 
   ```prediction: F      T 
                 747   1253
     label: F   [[602   398]    1000
            T    [145   855]]   1000
    Accuracy = 0.7285
    Precision = 0.6823623304070231
    Recall = 0.855 

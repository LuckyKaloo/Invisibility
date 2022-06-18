there is a lot of oop for some reason. 

The tf.functions will trace each time the objects are changed. idk how much of a performance difference it would make.
To make it faster change the object attributes to tf.Variable and use tf.Variable.assign
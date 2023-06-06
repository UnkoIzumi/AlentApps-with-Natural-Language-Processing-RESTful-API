import tensorflow as tf

hi = [1, 2, 3, 4, 5]
wr = [3, 5]

shape = (4, 10, 128)

x = tf.random.normal(shape)
print(x)

for w in hi:
    if w in wr:
        print(w)
    else:
        print("cannot")

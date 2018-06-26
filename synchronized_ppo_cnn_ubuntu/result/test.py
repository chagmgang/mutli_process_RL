import csv
import numpy as np

value = []
index = []

f = open('5process.csv', 'r')
csvReader = csv.reader(f)

for i, row in enumerate(csvReader):
    if i == 5000:
        break
    index.append(row[0].split()[1])
    value.append(row[0].split()[0])
    #if i % 2 == 1:
    #    index.append(row[0].split()[0])
    #    value.append(row[0].split()[1])

index = list(map(int, index))
value = list(map(float, value))

import tensorflow as tf

r = tf.placeholder(tf.float32) 
rr = tf.summary.scalar('time-consumed', r)
merged = tf.summary.merge_all()

sess = tf.Session()
writer = tf.summary.FileWriter('./board/process_5', sess.graph)

for i, values in enumerate(value):
    summary = sess.run(merged, feed_dict={r: values})
    writer.add_summary(summary, i)
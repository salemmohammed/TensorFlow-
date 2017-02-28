import tensorflow as tf
import argparse 
import sys
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


FLAGS = None

# Set parameters

learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

##################################################

# cluster specification
# we call this a set of jobs 
def main(_):

	parameter_servers = FLAGS.parameter_servers.split(",")
	workers = FLAGS.workers.split(",")

	cluster = tf.train.ClusterSpec({"ps":parameter_servers,"worker":workers})

	#print(parameter_servers,workers)

	# This for assign task for the job 
	server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

	#################################################

	# block and wait for connection to come in from nodes in the cluster
	
	if FLAGS.job_name == "ps":
		server.join()
	# server represent any particular task on the cluster 
	elif FLAGS.job_name == "worker":
		
		#cpu = FLAGS.cpu
		# number of Replica

		with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):

			with tf.name_scope("weights"):
				weights_1  = tf.Variable(tf.zeros([784,100]))
				weights_2  = tf.Variable(tf.zeros([100,10]))
			
			with tf.name_scope('bias'):
				bias_1 = tf.Variable(tf.zeros([100]))
				bias_2 = tf.Variable(tf.zeros([10]))

			# input the image 
			with tf.name_scope('input'):
				examples = tf.placeholder(tf.float32,[None,784])
				Labels   = tf.placeholder(tf.float32,[None,10])

			with tf.name_scope('Softmax'):
				first = tf.add(tf.matmul(examples,weights_1), bias_1)
				second = tf.nn.sigmoid(first)
				third = tf.add(tf.matmul(second,weights_2), bias_2)
				Estimates = tf.nn.softmax(third)


			with tf.name_scope("cross_entropy") as scope:
				cross_entropy = -tf.reduce_sum(Labels * tf.log(Estimates))
				tf.summary.scalar("cross_entropy",cross_entropy)

				# Optimization or training the model 
				# optimizer is the traiing. simply i train it to become good and less error 

			with tf.name_scope("train") as scope:

				# Execution the graph in the session 
				# TODO what is the Distributed Version 
				# count the number of updates

				global_step = tf.contrib.framework.get_or_create_global_step()
				train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

			with tf.name_scope("Accuracy"):
				predictions = tf.equal(tf.argmax(Estimates, 1), tf.argmax(Labels, 1))
				# Calculate accuracy
				accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

		# The StopAtStepHook handles stopping after running given steps.
		hooks=[tf.train.StopAtStepHook(last_step=10000)]
		print("Here-------------------------")

		# The MonitoredTrainingSession takes care of session initialization,
		# restoring from a checkpoint, saving to a checkpoint, and closing when done
		# or an error occurs.
		with tf.train.MonitoredTrainingSession(master=server.target,is_chief=True,checkpoint_dir="/tmp/train_logs",hooks=hooks) as mon_sess:
			print("Here-------------------------")
			while not mon_sess.should_stop():
				mon_sess.run(train_op)
		

		################################################



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true") # ??

	# Flags for defining the tf.train.ClusterSpec
	parser.add_argument("--parameter_servers", type=str, default="", help = "Comma to make saparable between jobs")
	parser.add_argument("--workers", type=str, default="" , help = "Comma to make saparable between jobs")
	parser.add_argument("--job_name", type=str , default="", help = "ps, or worker")

	# Flags for defining the tf.train.Server
	parser.add_argument("--task_index", type=int, default=0, help = " index of the task within the Job")
	parser.add_argument("--cpu", type=int, default=0, help = " cpu number ")
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
# Dependencies
import os, sys, re
import tensorflow as tf
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import vgg16
import vgg16HvassLabs
from IPython.display import Image, display # Not needed here
import time


# What is style transfer?? 
# Filtering or changing the images to match a filter based on visual properties, but subject to some optimization problem
# General knowledge from GOD SIRAJ: One shot or zero shot learning is the next step in ML and AI in general
# In style transfer, the computations outputted at primitive layers of a NN are used to evaluate a bunch of style losses
# Also, the outputs of the later deeper layers is used to compute the content losses. These are used as optimization inputs to tweak the image
# This is one of the approaches to the image-style transfer. Many other advanced approaches exist.


# Just check the version. It is 1.3.0 for me
#print(tf.__version__)


# Download the pre-trained VGG16 model
vgg16HvassLabs.maybe_download()


# Define function to load images
def LoadImage(name, max_size = None):
	# Open the image using PIL
	image = PIL.Image.open(name)
	# If we want to resize, pass max_size as the argument
	if max_size is not None:
		# Find out the factor with which the image needs to be scaled
		scaling_factor = float(max_size)/np.max(image.size)
		# Find the integer values of the image shape
		size = scaling_factor*np.array(image.size)
		print size
		size = size.astype(int)
		# Reshape
		image = image.resize(size, PIL.Image.LANCZOS)
	# Return as float32 the image
	return np.float32(image)


# Define a function to save an image
def SaveImage(im, name):
	# Clip the image values from 0 to 255
	im = np.clip(image, 0.0, 255.0)
	# Convert to uint8 format
	im = im.astype(np.uint8)
	# Save
	f = open(name, 'wb')
	PIL.Image.fromarray(image).save(file, 'jpeg')
	f.close()


# Define a function to display images
def PlotImage(im):
	# Clip pixel values
	im = np.clip(im, 0.0, 255.0)
	# Convert to uint8
	im = im.astype(np.uint8)
	# Display
	display(PIL.Image.fromarray(im))


# Define a function to plot the style transfer result
def PlotStyleTransferImage(content, style, mixed, is_sinc):
	# This should give a good introduction to plt as well!! BONUS
	#fig, axes = plt.subplot(1, 3, sharey = True)
	# Adjust the spacing
	#fig.subplots_adjust(hspace = 0.2, wspace = 0.2)
	# Define the interpolation
	if is_sinc:
		interp = 'sinc'
	else:
		interp = 'nearest'
	# Plot content
	this_fig = plt.figure()
	plt.subplot(131)
	plt.imshow(content/255.0, interpolation = interp)
	#im_ax.set_xlabel("Content Image")
	# Plot style
	plt.subplot(132)
	plt.imshow(style/255.0, interpolation = interp)
	#im_ax.set_xlabel("Style Image")
	# Plot mixed
	plt.subplot(133)
	plt.imshow(mixed/255.0, interpolation = interp)
	#im_ax.set_xlabel("Mixed Image")
	# Remove ticks
	#for im_ax in axes.flat:
	#	im_ax.set_xticks([])
	#	im_ax.set_yticks([])
	# After all the grooming, show the plot
	plt.show()
	# Wait for the beautiful face!!
	time.sleep(5)
	plt.close(this_fig)


# Define a function to compute Euclidean Loss between two feature maps (or, the mean squared error)
def EuclideanLoss(feat1, feat2):
	return tf.reduce_mean(tf.square(feat1 - feat2))


# Define a function to evaluate the content loss!!
def ContentLoss(sess, model, content, layer_ids):
	# sess: tf.Session(), model: vgg16 model, content: content image, layers: layers from which content needs to be fetched
	# The feed dictionary that we wish to use
	feed_dict = model.create_feed_dict(image = content)
	# Get pointers to tensors
	layers = model.get_layer_tensors(layer_ids)
	# Get the heat maps from the layers for the dictionary that we feed
	heat_maps = sess.run(layers, feed_dict= feed_dict)
	# Set model's graph as default. Not clearly documented it seems
	with model.graph.as_default():
		layer_losses = []
		for val, layer in zip(heat_maps, layers):
			# Define the tf constant and compute loss
			value_const = tf.constant(val)
			loss = EuclideanLoss(layer, value_const)
			# Append the loss to the list of losses
			layer_losses.append(loss)
		# Evaluate net loss
		net_loss = tf.reduce_mean(layer_losses)
	return net_loss


# Define a function for computing channel correlations using Gram Matrix
# Multiply a matrix by its transpose gives a gram matrix
def GramMatrix(tensor):
	# Get shape in the form of a list
	shape = tensor.get_shape().as_list()
	# Get number of channels: 4th entry
	channel_num = int(shape[3])
	# Define the matrix with other than channel dimensions flattened
	mat = tf.reshape(tensor, [-1, channel_num])
	# Define the gram matrix and return
	return tf.matmul(tf.transpose(mat), mat)


# Define function for style loss
def StyleLoss(sess, model, style, layer_ids):
	# sess: tf.Session(), model: vgg16 model, style: content image, layers: layers from which content needs to be fetched
	# The feed dictionary that we wish to use
	feed_dict = model.create_feed_dict(image = style)
	# Get pointers to tensors
	layers = model.get_layer_tensors(layer_ids)
	# Now, iterate over all the losses
	with model.graph.as_default():
		# Computer Gram matrix of all layers
		gram_layers = [GramMatrix(layer) for layer in layers]
		# Compute the values of gram_layers by passing feed_dict	
		layer_losses = []
		values = sess.run(gram_layers, feed_dict = feed_dict)
		# Compute the loss of layers
		for val, gram_layer in zip(values, gram_layers):
			value_const = tf.constant(val)
			loss = EuclideanLoss(gram_layer, value_const)
			layer_losses.append(loss)
		net_loss = tf.reduce_mean(layer_losses)
	return net_loss



# Define denoising method after mixing
def DenoiseLoss(model):
	return tf.reduce_sum(tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :])) + tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :])) 


# The image style transfer algorithm!!
def StyleTransfer(content, style, content_layers, style_layers, content_weight = 1.5, style_weight = 7.5, denoise_weight = 0.3, itr =500, step = 10):
	# Create a model instance
	model = vgg16HvassLabs.VGG16()
	# Create a session
	sess = tf.InteractiveSession(graph = model.graph)
	# Print content layer ids
	print("Content layer ids : " + str(model.get_layer_names(content_layers)))
	# Print style layer ids
	print("Style layer ids : " + str(model.get_layer_names(style_layers)))
	# Content Loss
	content_loss = ContentLoss(sess, model, content, content_layers)
	# Style Loss
	style_loss = StyleLoss(sess, model, style, style_layers)
	# Denoise Loss
	denoise_loss = DenoiseLoss(model)
	# Define adjusting variables
	content_adjust = tf.Variable(1e-10, name = 'content_adjust')
	style_adjust = tf.Variable(1e-10, name = 'style_adjust')
	denoise_adjust = tf.Variable(1e-10, name = 'denoise_adjust')
	# Initialize and run
	init = tf.initialize_all_variables()
	sess.run(init)
	# Optimize for the adjust variables
	sess.run([content_adjust, style_adjust, denoise_adjust])
	# New thing to learn!! Assign to adjust variables
	update_content_adjust = content_adjust.assign(1.0 / (content_loss + 1e-10))
	update_style_adjust = content_adjust.assign(1.0 / (style_loss + 1e-10))
	update_denoise_adjust = content_adjust.assign(1.0 / (denoise_loss + 1e-10))
	# Define combined loss
	combined_loss = content_weight*content_adjust*content_loss +  style_weight*style_adjust*style_loss + denoise_weight*denoise_adjust*denoise_loss
	# Define the gradients of combined loss with respect to model.input
	grad = tf.gradients(combined_loss, model.input)
	# Define the list of variables to run for
	run_for = [grad, update_content_adjust, update_style_adjust, update_denoise_adjust]
	# Initialize the mixed image as pure noise
	im_mixed = np.random.rand(*content.shape) + 128
	# Iterate
	for i in xrange(itr):
		feed_dict = model.create_feed_dict(image = im_mixed)
		grad_, content_adjust_val, style_adjust_val, denoise_adjust_val = sess.run(run_for, feed_dict = feed_dict)
		grad_ = np.squeeze(grad_)
		step_size_scaled = step/(np.std(grad_) + 1e-8)
		im_mixed -= grad_*step_size_scaled
		im_mixed = np.clip(im_mixed, 0.0, 255.0)
		print(". ")
		if (i%10 == 0):
			print('Iteration : ' + str(i))
		if (i == itr - 1):
			print('Iteration : ' + str(i))
			PlotStyleTransferImage(content, style, im_mixed, True)
	print('Final image : ')
	PlotImage(im_mixed)
	sess.close()
	return im_mixed


# Run the main
content = "./Me.jpg"
content_im = LoadImage(content, 300)
style = "./style_astro_1.jpg"
style_im = LoadImage(style, 300)
content_ids = [4, 5, 7, 9, 12]
style_ids = [1, 3, 6]
style_transfered_image = StyleTransfer(content_im, style_im, content_ids, style_ids, 10, 7.5, 0.3, 100, 10)

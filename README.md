# CSC3032-Major-Project-in-Computer-Science
## Overview
For this project I collaborated with researchers from Newcastle University and the Diego Portales University in Chile to develop a machine learning solution to assist their research
The problem was that creating the large number of images needed was too slow.
To solve this problem a Generative Adversarial Network (GAN) was developed to generate the images.
For this I first created the training data needed to train the GAN.
Next, I created the GAN, which required extensive research into relevant literature and research papers before I was able to develop a model that produced acceptable results. To improve on this research was conducted into multiple different GAN architectures which led me to choose a Deep Convolutional Generative Adversarial Network (DCGAN) as the improved solution.
The DCGAN was able to produce results which were indistinguishable from the training data.
### Glossary
- GAN = A machine learning model composed of a generator which generates images and a discriminator which aims to label images as real or generated.
These two models compete in order to improve the quality of the generated images.
- DCGAN = A GAN that makes use of deep convolutional layers.
- FCGAN = A GAN that makes use of fully connected layers.
## Generating the training data
For this project I had to generate the training data myself.
This involved using the Python package [LOICA](https://github.com/RudgeLab/LOICA).
I used this Python package to simulate the repressilator gene network and produce kymograph images.
By changing the values used different images could be produced.
I then generated two datasets of 10,000 images that were each simulated with different values for each image.
One dataset of images 28x28 in resolution and another of 72x72 in resolution.
The main idea was to use the lower resolution dataset for testing as training times would be less, and then the changes can be tested on the higher resolution dataset so that higher quality images can be generated.
The simulated images were .jpg images, I created a program to convert these images to .npy files which can be processed much faster.
## GAN selection
I knew that I would first develop a FCGAN because this would be the easiest to implement, but would not produce the best results.
Once a working FCGAN had been produced the knowledge gained would help me to develop a more complex GAN that would generate better quality images.

A few different GAN architectures were considered.
These included:
- CycleGAN:
This GAN is used to make modifications to already existing images.
For example you can train a CycleGAN to change images of horses into images of zebras.
This architecture was found to not fit my application so other options were considered instead.
- StyleGAN:
StyleGAN is a GAN that was developed by NVIDIA, because of this it has been trained extensively to generate high quality images.
StyleGAN2 made a number of improvements on the original which allowed for even better images to be generated.
This approach was considered but it would not have been complex enough for this project, because the model just needs to be imported and trained on my datasets to generate kymograph images.
This would have limited the number of findings I would have been able to make when compared to developing my own so this approach was also not chosen.
- DCGAN:
The final approach that was considered was to use a DCGAN.
This approach was shown to generate much higher quality images than a FCGAN in papers that I had reserched, so this approach was chosen.
## DCGAN development
The development of this model took much longer than the development of the FCGAN models.
This was because convolutional layers allow for much more configuration than the dense layers used in a fully connected model.
After research into methods to improve the quality of the generated images from a DCGAN and a large amount of my own testing the models were finished.
I then compared the images from the DCGANs to the FCGANs and it was clear that the image quality was much higher.
After enough training on the DCGANs the generated images were almost imperceptible to the training images.
## Assessing the quality of the generated images
How to assess the quality of generated images is a very complex problem and area of a lot of research in the GAN space.
This is because it is difficult to obtain a accuracy or quality value from a generated image so other methods need to be used.
I reviewed a number of metrics that could have been used but decided to use the Frechet Inception Distance (FID).

### FID
This method uses the pre-trained InceptionV3 model which is trained on 1,000 different classes of images.
The InceptionV3 model is given an image and the likeliness that that image belongs to each of the 1,000 classes is recorded.
This is done for thousands of training images and thousands of generated images.
The FID value can then be computed which represents how far the generated images are from the training images.
A FID score of zero means that the training images and the generated images are exactlty the same.
The calculated FID scores were 5-10x smaller for the DCGAN models which shows that they are generating much higher quality images.
## Reflection
Overall this project was a success because a GAN was produced that can generate images similar in quality to the training images.
The generated images were of a still relatively low resolution of 72x72 so future work could look at generating higher resolution images.
This would likely require more than the 10,000 training images that I used for training and would require more time to train the models.
Alternative GAN architectures could also be the topic of a future project.
Of the alternatives I outlined in the final report I think the most promising would be to make use of NVIDIA's StyleGAN for transfer learning.
StyleGAN is capable of generating high-quality high-resolution images so a transfer learning approach may allow for the same results but for this dataset.

This project allowed me to increase my knowledge of machine learning specifically with image generation as well as data science methods that were needed to process the large amount of training data.

# Image Colorizer

Python program that converts a Black & White image to Colored. Uses various ML algorithms such as SVR to predict the color of each pixel. 

![Demo](https://i.imgur.com/bFGoxCj.jpg)

## Setup

1. <b>Install Necessary Packages</b>
   
    All the necessary packages can be installed by using pip:
    
    `pip install -r requirements.txt`
    
2. <b>Download Dataset</b>
   
    The dataset required to train the model can be created with the help of the web scraper included.
    
    `python get-images.py`
    
    The script is designed to go on an unlimited loop, terminate the script when necessary amount of images have been downloaded.
    
3. <b>Train the Model</b>

    The model can be easily trained by running the train-model script
    
    `python train-model.py`

4. <b>Test the Model</b>

    In order to test the model you need to pass a black and white as input:

    `python test-model.py <image_name>`

## Process Explanation

### Training The Model

1. Generating Sub squares -

   For a given image generate the following -

   - Array of subsquares

   - Array containing average U values for the sub squares

   - Array containing average V values for the sub squares


   1. Segmentation - 

      Segmentation is done to change the representation of the image into something that is more meaningful and easier to analyze.

      Functions used - 

      ```python
      skimage.util.img_as_float()
      skimage.data.imread()
      skimage.segmentation.slic()
      ```

   2.  Get YUV Values - 
      From the RGB values that were found from the segmentation process calculate the YUV values using the conversion matrix.
    
        [Formula Explanantion](https://www.pcmag.com/encyclopedia/term/55166/yuv-rgb-conversion-formulas)
    
        Use `np.dot()`to calculate [dot product](https://www.tutorialspoint.com/numpy/numpy_dot.htm).

   3. Find n_segments -

      The N segment of the image will be the maximum value in the segments array plus 1.

   4. Computing Centroids -

      Create  templates of U, V, centroids and point_count using [np.zeros()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndenumerate.html)

      Iterate through the segments array with the help of [np.ndenumerate()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndenumerate.html) and calculate each element in the arrays.

      Calculate Centroids, U and V with the help of point_count.

   5. Calculate the Sub Square - 

      Create a template for the Sub Square.

      Calculate the sub square using fourier transform with the help of [np.ftt.ftt2](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html)

2. Train Models - 

    1. Training the base model - 
       
        Using [sklearn.svm.SVR()](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) train the base models using [SVM](https://cs.adelaide.edu.au/~chhshen/teaching/ML_SVR.pdf) with constant values of C and Epsilon.

    2. Fitting the Model - 

        Using the [fit()](http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.fit.html) function, [fit](https://www.quora.com/What-does-fitting-a-model-mean-in-data-science) the base model using X, U_L and V_L.

## Checklist:

Modules Involved -

- <del><b>get-images</b> -- Design a web scraper to  get images of landscapes that will end up becoming the data set.  </del>
- <del><b>train-model</b> -- Utilize Support Vector Regressions to segment and train on the dataset based on YUV colorspace.</del>
- <del><b>test-model</b> -- Use multiple layers of SVR on test image then apply Markov Random Fields to find hidden chrominance values and generate an output.</del>  


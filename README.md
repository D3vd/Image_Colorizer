# Image Colorizer

## Checklist:

Modules Involved -

- <b>get-images</b> -- Design a web scraper to  get images of landscapes that will end up becoming the data set.  
- <b>train-model</b> -- Utilize Support Vector Regressions to segment and train on the dataset based on YUV colorspace.
- <b>test-model</b> -- Use multiple layers of SVR on test image then apply Markov Random Fields to find hidden chrominance values and generate an output.  



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
    
        Using the [fit()]() function, [fit](https://www.quora.com/What-does-fitting-a-model-mean-in-data-science) the base model using X, U_L and V_L.
    
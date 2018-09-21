# Image Colorizer

## Checklist:

Modules Involved -

- <b>get-images</b> -- Design a web scraper to  get images of landscapes that will end up becoming the data set.  
    
- <b>train-model</b> -- Utilize Support Vector Regressions to segment and train on the dataset based on YUV colorspace.
 
- <b>test-model</b> -- Use multiple layers of SVR on test image then apply Markov Random Fields to find hidden chrominance values and generate an output.  
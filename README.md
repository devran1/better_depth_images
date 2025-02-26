
### Making depth maps more visible
Number of objects inside the image is used to create bins for discretization. The bins are used to create a new image, then the new image is given to the AI (intel-isl/MiDaS DPT_Large) for depth map generation. This process is repeated many times for really bad depth images, on the other images it is repeated only once. We used Amazon Armbench dataset.


<a href="url"><img src="evidence3.png" align="left"></a>

--------

<a href="url"><img src="evidence2.png" align="left"></a>

--------

<a href="url"><img src="evidence.png" align="left"></a>

--------
We also added Gaussian, bilateral and medium blur before the depth maps, and we added a simple quality enhancement but the results are not that impressive images can be found inside the folders.

### amazon armbench dataset 
http://armbench.s3-website-us-east-1.amazonaws.com/segmentation.html
### The article
https://www.amazon.science/blog/amazon-releases-largest-dataset-for-training-pick-and-place-robots


##### Some Sources That I Used.
https://stackoverflow.com/questions/14947909/python-checking-to-which-bin-a-value-belongs

https://www.makeuseof.com/opencv-image-enhancement-techniques/

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Ryan Morris
  * [LinkedIn](www.linkedin.com/in/r19)
* Tested on: (TODO) Windows 11, i7-12700H @ 2.3GHz 64GB, GeForce RTX 3070 Ti Laptop GPU 8GB

**Note for graders**: Submission version for grading 9/7 11:59PM (1 late day used)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

## Performance Analysis (Wip)

- For each implementation, how does changing the number of boids affect performance? Why do you think this is?
- For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
- For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
- Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!


Step 1: Create graph with framerate change with increasing boid # for each solution, one for each with and without visualization
Step 2: Framerate change for increasing block size (this is the 128 thing), I guess can run this on best solution
Step 3:
Step 4: Show bloopers, etc. 

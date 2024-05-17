Nicolas HAMMJE 28710225 Haitam RARHAI 28712747

# ROS Project README

## Prerequisites

To run our project, you need to install some Python libraries:

- scikit-learn
- scikit-image
- numpy
- scipy

You can install all the required libraries using the following commands:

pip install -r requirements.txt

## Setting Up the Parameters

Before running the project, you can set the order of gates (bottles) that the TurtleBot will navigate through. It is recommended not to place the gate that is the furthest in the first position, as it would require the TurtleBot to navigate around all the other gates to reach it.

To set the order, change the "order" param in the simu_params file in the /params folder. There are two files, for the two worlds. The colors are:

0: Blue
1: Green
2: Yellow/Red

or set the params using rosparam set while the robot is running. 

## Launching the Simulation

After setting the parameters, you can launch the project. This will start Gazebo, initialize the TurtleBot, and autonomously navigate through the circuit.

Use the following command to launch the project:

```
roslaunch projet challenge1.launch
```

## Recommendations

We recommend using `challenge1` with the first world setup. This setup has specific parameters optimized for the distances between the gates in the first world. The second world (`challenge2`) has different distances between the gates, which might affect the performance of the TurtleBot.

On our computers, the TurtleBot completed all challenges of the first world flawlessly. The second world is a bit more complicated, and is for some reason non-deterministic, as in multiple launches with no changes result in different outcomes. 

We also note that if the computer running the process is too slow, the success rate is going to be severely hampered (Monitor the "real time factor" in Gazebo). While not terribly inefficient, RRT* uses a lot of ressources, and failure to compute paths in a relatively short amount of time can result in collisions. We have tested the current code on a MacBook Pro M1 Max running Ubuntu 20.04 in Parallels. 
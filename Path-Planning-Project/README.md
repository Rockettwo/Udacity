# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program
   
### Simulator.
You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).  


### Goals
In this project the goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. The car's localization and sensor fusion data is provided, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, while other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

## Solution

The solution is based on a cost minimizing trajectory planning. For each time step four different trajectories are calculated. These consist of one trajectory for each target lane and one more to stay in the current lane and slow down.
The trajectories are evaluated considering collisions only, as they already are velocity, acceleration and jerk optimized. The preferred lane is the middle lane.

There is a state if the vehicle is following another vehicle. Only if another faster trajectory is having no collision for 150 consecutive times, the following state is escaped.

### Structure
All code is included in _main.cpp_ where modifications and introductions of global variables were made in lines 40-43. The path planning algorithm starts at line 110 and finishes at line 287.
It can be roughly separated in three parts.

##### Preprocessing of the previous trajectory and determining current state (line 115 - 163)
During this step a specific amount of data points from the last trajectory is taken as a starting point for the new trajectory.

##### Generation of the trajectories (line 165 - 274)
In this step the different trajectories are calculated based on the same base. The different trajectories are:
- Stay on same lane and reduce speed to fit car driving in front.
- Calculate with target on left lane
- Calculate with target on middle lane
- Calculate with target on right lane

This means that changes from left to right are also possible. Also, no information about the current position is needed for the last three trajectories.

##### Evaluation and selection of trajectory (cost included in above lines 245 - 275 | selection 276ff.)
In this step the cost function for each trajectory is calculated. If a car has a collision with another predicted car, 11000 (random number) is added to the cost of this trajectory. 

If there is no valid trajectory, the follow trajectory is always chosen. It also switches state to following then. This means, that some calculations are changed, e.g. speed is lowered each time until the follow speed is reached.

Normal operation is achieved once there is a trajectory with no collision for at least 50 consecutive plannings.

After the cost calculation the trajectories are sorted and the trajectory with the lowest cost is chosen.

##### Remarks

All in all this approach works quite well and doesn't need any explicit state machine. Yet in some cases it might fail to comply with the maximum jerk and acceleration when it goes over two lanes. It should never have a collision when it is avoidable.

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

---

## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

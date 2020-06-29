# PID-Control-Project

## Implementation

The PID controller is implemented in the given framework. A feature to limit the pid return values was added.

## Reflection

### Proportional part of controller (P)

The P-term is using the absolute error directly. Thus, it is prone to overshoot (if Kp is set to high) or slow convergence (if Kp is set to low).
This is usually the starting point for the manual parameter tuning or initial parameter finding. It is easier to adjust Kp to avoid the two aforementioned drawbacks and get good behavior.

### Integral part of controller (I)

The P-term alone can't achieve zero error, because there will always be a little overshoot. The I-term is solving this problem. It tries to decrease the cumulated error which also means no static error. 
The main problem using an I-part is that the cumulated sum may actually cause the Controller to drift away, because it is getting larger and larger.

To address this problem, Ki is chosen to be a very small value. This reduces the impact of this part. Also the current system does not have any measurement errors which would also require a I-term to be included.

### Derivative part of controller (D)

The D-term's purpose is mainly to give faster corrections than P- or I-term does. It calculates the gradient and gives feedback according this gradient which makes it suitable for dynamic situations.

If Kd is too high, it may easily cause the system to swing up. In my opinion this is the most critical parameter to chose.

### Choice of parameters

I chose manual tuning, because it seemed to work pretty well and fast. PID's usually end up requiring some manual attention in the end anyways. 

My procedure for selecting the parameters was as follows (see Ziegler-Nichols method in [this tutorial](https://www.thorlabs.com/tutorials.cfm?tabID=5DFCA308-D07E-46C9-BAA0-4DEFC5C40C3E)):

1. Set all gains to zero. 
2. Increase the P-gain until a steady oscillation is reached. 
3. Measure the period of the oscillation.
4. Calculate Kp, Kd and Ki according to the tutorial.
5. Tune values

The values I got from the calculation were not usable, but after I lowered Ki a lot everything worked quite well. It seems that Ki in this application (probably because of the circular track) didn't have much positive impact.

My final values can be seen in `main.cpp`. 


### Speed controller

I designed a PID for the throttle without any advanced method.

The target speed is set to 50 mph if the cte is not too high and steering values are low. Else the target speed is set to 30 mph (mostly during curves).

### Improvements
The automation of the tuning process is definitely a big point. I've seen some good approaches using backpropagation or SGD. Maybe I'll implement these methods in the future.

Sequential Monte Carlo

<img src="https://latex.codecogs.com/gif.latex?X_{n&plus;1}&space;=&space;\Phi&space;X_n&space;&plus;&space;\Psi_z&space;Z_n&space;&plus;&space;\Psi_w&space;W&space;\\&space;Y_n^l&space;=&space;v&space;-&space;10\eta&space;log_{10}||(X_n^1,&space;X_n^2)^T&space;-&space;\pi_l||&space;&plus;&space;V_n^l"/>

Where X is a (6, 1) vector of a vehicle with position, velocities and acceleration, Z is a (2, 1) vector representating a force or acceleration of command at each time step to turn its handle to 4 directions (north, south, east and west) or do nothing and W is a noise term.
Y is a model for a signal value that the vehicle receives from 6 points (position, Pi) wiht some variables and nose term (V).

Aim to find the track of the vehicle only from the signal record that it receices every time step.

First, Sequential Importance Sampling is implemented since we know the smoothing function up to normalizing constant.
--Let g(X) and z(X) be a instrumental function and smoothing function without the denominator. If g(X) is taken so that important weights are written down easily, then an algorithm will be below

<img src="https://latex.codecogs.com/gif.latex?\omega_{n&plus;1}^i&space;=&space;\frac{z_{n&plus;1}(X_{0:n&plus;1})}{g_{n&plus;1}(X_{0:n&plus;1})}&space;=&space;P&space;\omega_n^i&space;\\&space;\tau[x_n|Y]&space;=&space;\sum_{i}&space;\frac{\omega_n^i}{\Omega}x_n^i"/>

where i is a index of one sample and n is an arbitrary time step.


Second, to overcome its particle degeneracy, SIS with Resampling is introduced. Every steps the samples are resampled from themselves according to important weights. Then, the algorithm is 

<img src="https://latex.codecogs.com/gif.latex?\omega_{n&plus;1}^i&space;=&space;\frac{z_{n&plus;1}(X_{0:n&plus;1})}{g_{n&plus;1}(X_{0:n&plus;1})}&space;=&space;P&space;\\&space;\tau[x_n|Y]&space;=&space;\sum_{i}&space;\frac{\omega_n^i}{\Omega}x_n^i"/>

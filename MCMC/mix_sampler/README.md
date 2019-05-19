Hybrid sampler

![coal_mine](https://github.com/washingtk/kth/blob/pictures/mcmc_pic/coal_mine.png)

To find a intimacy of a disaster in a certain inteval, set parametes as below

<img src="https://latex.codecogs.com/gif.latex?breakpoints&space;:&space;d&space;\\&space;time:&space;t_1=1855,&space;t_{d&plus;1}&space;=&space;1963,&space;\&space;1855&space;<&space;t_i&space;<&space;1963\\&space;intimacy:\lambda&space;\sim&space;\Gamma(2,&space;\theta)&space;\\&space;disastes\&space;in\&space;an\&space;interval&space;:&space;\tau_i&space;\\&space;\\&space;theta:\theta&space;\sim&space;\Gamma(2,&space;\nu)"/>

a few more assumptions with this yields a conditional probability

<img src="https://latex.codecogs.com/gif.latex?f(\theta|\tau,&space;\lambda,&space;t)&space;=&space;\Gamma(2&space;&plus;&space;2d,&space;\nu&plus;\sum&space;\lambda)&space;\\&space;f(\lambda|\theta,&space;\tau,&space;t)&space;=&space;\Gamma(2&space;&plus;n_i(\tau),&space;\theta&space;&plus;&space;t_{i&plus;1}-t_i)&space;\\&space;f(t|\lambda,&space;\theta,&space;\tau)&space;\propto&space;exp(\sim)\prod&space;\lambda^{n_i(\tau)}(t_{i&plus;1}-t_i)&space;\\"/>

applying Gibbs sampler to the first two and Matropolis Hastings sampler to the last.

result(matlab):
![breakpoints](https://github.com/washingtk/kth/blob/pictures/mcmc_pic/breakpoint4.png)

TODO:
numpy RUNTIMEWARING:encounter double scalar

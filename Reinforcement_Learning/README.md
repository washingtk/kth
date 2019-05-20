Blackjack

Finding an otimal policy of Blackjack by two methods of Reinforcement learning.
Settings :
<img src="https://latex.codecogs.com/gif.latex?Reward&space;=&space;\begin{cases}&space;1.5&space;&&space;\text{&space;if&space;blackjack}\\&space;1.0&&space;\text{&space;if&space;wins}&space;\\&space;-1.0&space;&&space;\text{&space;if&space;loses&space;}&space;\end{cases}"/>

Let S, A, R be a state space, an action space and a reward. Then, the state value (and action value) is defined by
<img src="https://latex.codecogs.com/gif.latex?Reward&space;=&space;\begin{cases}&space;1.5&space;&&space;\text{&space;if&space;blackjack}\\&space;1.0&&space;\text{&space;if&space;wins}&space;\\&space;-1.0&space;&&space;\text{&space;if&space;loses&space;}&space;\end{cases}" />
where t is a certain time step and pi is a policy.

If a policy is stationary one, the Bellman equaiton is
<img src="https://latex.codecogs.com/gif.latex?v_{\pi}(s)&space;=&space;E_\pi[R_{t&plus;1}&space;&plus;&space;\gamma&space;v_{\pi}(S_{t&plus;1})|S_t=s]"/>
The Bellman optimal equation is defined similarly for the optimal policy.

On policy Monte Carlo: see some refference

Q-larning:(based on action value)
<img src="https://latex.codecogs.com/gif.latex?q_t(S_{t-1},&space;A_{t-1})=q_{t-1}(S_{t-1},&space;A_{t-1})&space;&plus;&space;\alpha[R_t&plus;\gamma&space;\sup_{a&space;\in&space;A(S_t)}&space;q_t(S_{t},&space;a)-q_{t-1}(S_{t-1},&space;A_{t-1})]\\"/>
with epsilon greedy algorithm.


Result:(when the num of decks is inf)

![optimal policy](https://github.com/washingtk/kth/blob/pictures/pic/sum_Optimal%20Policy%20in%20infdeck.jpg)




#### Background

Examples of such extensions include the Multi-agent Markov Decision Process (MMDP)
proposed by Boutilier [3], the Partially Observable Identical Payoff Stochastic Game (POIPSG) proposed by
Peshkin et al. [10], the multi-agent decision process proposed by Xuan and Lesser [15], the Communicative Multiagent Team Decision Problem (COM-MTDP) proposed by
Pynadath and Tambe [11], the Decentralized Markov Decision Process (DEC-POMDP and DEC-MDP) proposed
by Bernstein et al. [2], and the DEC-POMDP with Communication (Dec Pomdp Com) proposed by Goldman and
Zilberstein [5].

[2] Daniel S. Bernstein, Robert Givan, Neil Immerman, and
Shlomo Zilberstein. The complexity of decentralized control of Markov decision processes. Mathematics of Operations Research, 27(4):819–840, November 2002.
[3] Craig Boutilier. Sequential optimality and coordination in
multiagent systems. In Proceedings of the Sixteenth International Joint Conference on Artificial Intelligence, pages 478–
485, 1999.
[5] Claudia V. Goldman and Shlomo Zilberstein. Optimizing information exchange in cooperative multi-agent systems. In
Proceedings of the Second International Joint Conference on
Autonomous Agents and Multi Agent Systems, pages 137–
144, Melbourne, Australia, July 2003. ACM Press.


## RL
1. historical
2. RL as MDP
3. resolution methods through estimation of V/Q
tabular to DNN?
value estim? actor-critic?
4. MARL as MG
Nash equilibrium?
5. coop/coord/mixed
6. centralized or decentralized rewards MDP-dec
7. partial observability POMDP-dec
8. problem of npexp-complete resolution
9. introduction of communication/consensus
10. byzantine generals
11. interactive consensus

> For problem definition use [[arXiv:1912.00498v1]]
> For many definitons use [[/s10462-021-09996-w]]

(1.) 
Reinforcement learning is a type of learning in which an agent learns to perform actions that maximize a reward signal. It is inspired by the way we suppose learning naturally happen through rewards and punishmens and find it's roots in psychology literature [missing] and animal experimentations [pavlovian_experiment]. For example, dogs can be trained to search for hidden objects by rewarding them each time they locate the object. Based on this feedback, the dog will adapt its behavior and learn to search for objects on command. The goal of reinforcement learning is to discover a policy, or a set of rules that dictate how the agent should act in different situations, in order to maximize the rewards it receives [Sutton and Barto 1998]. These rewards can be either positive or negative, indicating a reward or punishment, respectively.
(2.)
The standard formulation for such sequential decision-making is the Markov decision process, $M = (S, A, P, r, γ)$, where:
- $S$ is the state-space $S = {1, 2, ... , |S|}$, a finite set of world states
- $A$ is the action-space $A = {1, 2, ... , |A|}$, a finite set of actions
- $P$ is the function of state transition probability $P(s'|s, a)$, that express the probability of transitioning from a state $s$ to $s'$, selecting action $a$
- $R$ is the reward obtained from the reward function $R$ given the transitioning of state $r = R(s, a, s')$, from a state to another.
- $γ \in [0, 1]$ is the discount factor to handle both finite and infinite-horizon problems (discount factor)

The agent's objective is to find a deterministic, optimal policy $\pi^∗ : S \rightarrow A$, that will dictate how to act in order to maximize its rewards
$π∗:= arg maxπ∈Θ E " ∞Xk=0γ^kr(s_k, π(s_k))$
$\pi^* = \arg \max_{\pi \in \theta} \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t R(x_t, \pi(x_t), x_{t+1}) \mid x_0 = x]$
where $\theta$ is the set of all admissible deterministic policies, and $(s_0, a_0, s_1, a_1, ...)$ is a state-action trajectory generated by the Markov chain under policy $π$.

(3.)
As an alternative to the direct search for the optimal policy, we define two utility functions that describe the concept of expected return:
- $V^π(s) = E [P ∞k=0 γkr(sk, π(sk))|s_0 = s]$, is the value function for a given policy $π$, which encodes the expected cumulative reward when starting in the state $s$, and then, following the policy $π$ thereafter.
- $Q^π(s, a) = E [P ∞k=0 γ^kr(s_k, π(s_k))|s_0 = s, a_0 = a]$, is the state-action value function or Q-function, which measures the expected cumulative reward when starting from state $s$, taking the action $a$, and then, following the policy $π$.
While those could be simply represented in a tabular format, in the context of deep reinforcement learning, either the policy, the value functions or both are represented by neural networks. [[more on that in 1912.00498v1 p.4]]

(4.)
Extending the single agent case to a multi agent context, we assume a different formlization which can be seen as a generalizition of MPDs and takes the name of Markov Game. There, each agent has its own policy for deciding which actions to take in each state, and the agents' policies may be different and may affect each other's rewards and future states. This allows for modeling more complex decision-making scenarios where agents need also to make strategic decisions adapting to each other behaviour. From [Littman 1994] this is formalized by the tuple $(N, X, {U_i}, P, {R_i}, \gamma)$, where:
- $N = \{1, …, N\}$ denotes the set of $N > 1$ interacting agents
- $O$ is the set of states observed by all agents
- $U = U_1 \times \cdots \times U_N$ joint action space is the collection of individual action spaces from agents $i \in N$
- $P: X \times U \rightarrow P(X)$ is the transition probability function and describes the chance of a state transition
- $R_i$ is the reward function defined as $R_i: X \times U \times X \rightarrow \mathbb{R}$ associated to each agent $i \in N$
- $\gamma \in [0,1]$ is the discount factor.

At stage $t$, each agent $i \in N$ selects and executes an action depending on the individual policy $\pi_i: X \rightarrow P(U_i)$. The system evolves from state $x_t$ under the joint action $u_t$ with respect to the transition probability function $P$ to the next state $x_{t+1}$, while each agent receives $R_i$ as immediate feedback to the state transition. The goal of each agent, similarly to a single-agent problem, is to modify its policy in order to maximize its long-term rewards. [[/s10462-021-09996-w]] Here, We can differentiate between settings in which the rewards obtained are shared or assigned individually and this is strictly linked to the type of problem. The relative taxonomy classify them as:

(5.)
- Fully cooperative setting, all agents receive the same reward $R = R_i = \dots = R_N$ for state transitions. In this setting, agents are motivated to collaborate in order to maximize the performance of the team. In general, we refer to cooperative settings when agents are encouraged to collaborate but do not have an equally-shared reward.
- Fully competitive setting, the problem is described as a zero-sum Markov Game where the sum of rewards equals zero for any state transition, i.e. $R = \sum_{i=1}^N R_i(x, u, x') = 0$. In this setting, agents are motivated to maximize their own individual reward while minimizing the reward of others. In a loose sense, we refer to competitive games when agents are encouraged to excel against opponents, but the sum of rewards does not equal zero.
- Mixed setting, also known as a general-sum game, the setting is neither fully cooperative nor fully competitive, and therefore does not impose any restrictions on the goals of the agents.

(6.) + (7.)
In addition to the reward structure, other taxonomies may be used to differentiate between the information available to agents. - Decentralized settings are the ones where independent learners are not aware of the existence of others and cannot observe their rewards and selected actions [[Bowling and Veloso (2002) and Lauer and Riedmiller (2000)]].
- Centralized settings present joint-action learners able to observe a-posteriori the taken actions of all other as shown in Hu and Wellman (2003) and Littman (2001).

For Indipendent learners one may imagine reinforcement learning algorithms running in N identical and independent simulated environments but with the environment evolving by unknown causes. This situation is defined as non stationarity or moving target problem since the change in transition probability function due to co-evolution of all agents' policy and assume the following formulation:

$P(s' \mid s, u, \pi_1, \dots, \pi_N) \neq P(s' \mid s, u, \bar{\pi_1}, \dots, \bar{\pi_N})$

Note that the form of rewards and information at disposal, makes a huge difference in multi-agent systems. If every agent receives a common reward and have a full knowledge of the environment transition (fully observable and fully cooperative), the situation becomes relatively easy to deal with, and agents can, in principle, learn a joint stochastic policy 
$π : S × U → [0, 1]$
that means we can reduce the multi-agent problem to a single-agent problem finding an exact optimal policies of the underlying decision problem even without coordination among agents [10].

However, usually these assumptions do not match the reality and even if that was the case we would be tempted to factorise the joint policy into $\pi(u \mid s_t) = \forall_a \pi_a (u_a \mid s_t)$, to avoid the exponential growth of the action space U with the number of agents, which may render greedy action selection, exploration and learning architectures intractable.
This leads to the problem of partial observability, in which the agents should act and learn haveing a limited knowledge of the world state.

(8.)
As this thesis is focused on such decentralised control settings under partial observability assumption is necessary to underline how the decision problem for Decentralised partially observable Markov decision processes (Dec-POMDPs) is known to be extremely challenging. In fact, computing even an approximately optimal policy to DEC-POMDPs is NEXP-complete (Rabinovich et al., 2003)[1]. Despite some recent empirical successes [2]–[4], finding an exact solution of Dec-POMDPs using RLs with theoretical guarantees remains an open question. [[arXiv:1912.00498v1]]
Nevertheless introducing the relaxation of free communication between agents, we expand the knowledge of the agents moving the problem in P-SPACE(Blondel and Tsitsiklis, 2000), without introducing unrealistic abstraction attainable only to simulations.

(9.)
Seeking for coordination between decentralized indipendent agents some sort of communications and agreement on action should be introduced. We can think this as a distributed optimization problem where we seek consensus in the policy development (in the development of an optimal policy attanaible to multi agent contexts) through only local computation and communication with neighboring agents.

(10.)
In standard consensus algorithm [[arXiv:1912.00498v1]] we have a set of agents $a_i$, from $A=\{i \in 1, 2, \dots, N\}$, each initialized with some initial value. To be able to communicate we can think the agents being interconnected over an underlying reliable communication network ideally represented as an oriented graph. To reach consensus, every agent communicate with one another, exchanging values, locally process the information and then proposes a single value $v_i$, drawn from the set $R=\{i \in 1,2, \dots, M\}$. The agents are said to reach a consensus if from a certain timestep $t$ it holds,
$\lim_{t \to \infty} v_1^t = v_2^t = \dots = v_N^t, \quad \forall i \in V$, for every set of initial values ∈ R^m.

A consensus algorithm is considered to be formally correct [[Colouris]] if it satisfies the following three conditions in every execution:
1. Termination: Eventually, all correct processes set their decision variable.
2. Agreement: The decision value of all correct processes is the same.
3. Integrity: If all correct processes propose the same value, then any correct process in the decided state must choose that value.

Keeping that in mind but relaxing some strict constraint like the convergence to a single state value of consensus we can design a method of agreement between agents to optimize the local policy conformly to the necessity of the group 

<!--Transposing such idea to a step of reinforcemnt learning policy agreement we can think the proposed value $v_i$ as a representation of the intention of the agent in the continuos space and relaxing some requirements is it possible to design a method which optimize such representation-->

<!--
# TO add.
In contrast, a lot of work has focused on understanding agents’ communication content; mostly in discrete settings with two agents (Wang et al., 2016; Havrylov & Titov, 2017; Kottur et al., 2017; Lazaridou et al., 2017; Lee et al., 2018). Lazaridou et al. (2017) showed that given two neural network agents and a referential game, the agents learn to coordinate. Havrylov & Titov (2017) extended this by grounding communication protocol to a symbols’s sequence while Kottur et al. (2017) showed that this language can be made more human-like by placing certain restrictions. Lee et al. (2018) demonstrated that agents speaking different languages can learn to translate in referential games.
-->
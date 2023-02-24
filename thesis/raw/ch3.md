
## Related Works

come affronti l'environment
1. approaches to large space problems (like use of trajectories and IL)
come affronti il learning
2. decentralization of the work
3. CTDE (and why we focus on it)
come comunicano
4. coordination through communication with gradients of implementation details
5. review of methods of communication
6. communication trough observations as shared knowledge (and why we focus on it)
come prendono in considerazione le informazioni provenienti dagli altri
7. methods of integrate the knowledge and make decisions (promise of explanation of the method in the section)


<!--
Sutton, R.S.; Barto, A.G. Reinforcement Learning: An Introduction; MIT Press: Cambridge, MA, USA, 2018

### (1a)
- **Simulation-Based Reinforcement Learning for Real-World Autonomous Driving**
sim-to-real policy transfer ( RGB + segmentation mask)

### (1aa)
- **Tackling Real-World Autonomous Driving using Deep Reinforcement Learning**
synthetic environment simple (4stack TOP camera + info auto) doppia net per prediction of policy values distribution
modulo deep-model per gesstire delay between synth training to real
first pretraining with IL then RL in synth and test in real

### (1b)
- **Data-Efficient Hierarchical Reinforcement Learning**
Claim of inefficency in HRL solvable with a more general approach and working off-policy. "Having a hierarchy of policies, of which only the lowest applies actions to the environment an the higher one instead predict for long term strategies"
They predict goals with higher level policy and reward the lower level policies when they do actions able to lead to the gaol. They used a modified experience replay (update the goal in the samples collected). The reward function is parametrized as well by the current goal
BAD: (unlike others that works in the embedded space) they reward states by matching raw observations and that could be expansive in complex environment, they claim usable for navigation but it is used in simple env and is focused to robotic applications where you have position vectors.
GOOD: HPC implementation with goal directed command is good to make complex plans
Is it well suited for multi agent interacting?
It's stated by authors that the solution is not so stable and that the architecture can be improved

### (1c)
- "Imitation Learning: A Survey of Learning Methods" by Ahmed Hussein, Mohamed Medhat Gaber, Eyad Elyan, Chrisina Jayne (https://doi.org/10.1145/3054912)

### (1d)
Hutsebaut-Buysse, M.; Mets, K.; Latré, S. Hierarchical Reinforcement Learning: A Survey and Open Research Challenges. Mach. Learn. Knowl. Extr. 2022, 4, 172-221. https://doi.org/10.3390/make4010009

### (1e)
"A Survey of Scalable Reinforcement Learning" by George B. Stone, Douglas A. Talbert, William Eberle from International Journal of Intelligent Computing Research (IJICR), Volume 13, Issue 1, 2022

### (1f)
A Survey on Intrinsic Motivation in Reinforcement Learning by Arthur Aubret, Laetitia Matignon, Salima Hassas. arXiv, 2019. https://arxiv.org/pdf/1908.06976.pdf

### (1g)
Multi‑agent deep reinforcement learning: a survey

### (1h)
A Survey and Critique of Multiagent Deep Reinforcement Learning

-->

## Possible methods
Multi-Agent Reinforcement Learning (MARL) involves training multiple agents to learn and interact in a shared environment. Handling large environments in MARL can be challenging due to the added complexity of having multiple entities interacting and the necessity of adopting scalable approaches [1g,1h,1e]. A reliable solution to handle such problems is to use imitation learning [1c], in where agents learn to imitate the actions of an expert demonstrator, that a mapping from colected observations to actions, in order to learn how to explore the environment effectively. For example [1a,1aa] uses this approach to transpose the driving ability acquired by human experts into an agent policy able to control a physical car, therefore avoiding the extensive trial and error exploration of the enourmous state space which characterize autonomous driving. This approach decreases the training time by providing better example, however, you'd require a lot of expert trajectories that are strictly dependent on the specific characteristics of the environment and the task at hand therefore not reusable in other applications, but also during training there is an important potential for overfitting on the trajectories collected.
Another relevant method is hierarchical reinforcement learning (HRL) [1d,f]. In HRL, the learning process is divided into multiple levels or layers, with each level representing a different level of abstraction or granularity in the task. (For example, an RL agent learning to play a video game might have one level that focuses on low-level actions such as moving and jumping, and another level that focuses on higher-level strategies such as exploring the environment or attacking enemies.) By decomposing the learning process into multiple levels, HRL can help to simplify the learning process and make the problem more manageable: indeed is very often applied to robotic control problems [1b]. Nevertheless it can be challenging to implement, as it requires careful design of the hierarcal structure and of abstraction at each level: especially in MARL contexts too high abstraction let arise important problems like remaining stuck during training in suboptimal solutions, while too little abstraction can result in a lack of generalization.


<!--
### (2a)
[Papoudakis et al., 2021] Georgios Papoudakis, Filippos Christianos, Lukas Sch¨afer, and Stefano V Albrecht. "Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks". In NeurIPS, 2021.

### (2b)
Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments". In Advances in Neural Information Processing Systems, pages 6379–6390, 2017

### (2c)
Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. "Counterfactual multi-agent policy gradients". In Counterfactual Multi-Agent Policy Gradients, 2018.

### (2d)
Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore Graepel. "Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward". In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS ’18, pages 2085–2087, Richland, SC, 2018. International Foundation for Autonomous Agents and Multiagent Systems. event-place: Stockholm, Sweden.

### (2e)
Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, and Yung Yi. Qtran: learning to factorize with transformation for cooperative multi-agent reinforcement learning". In International Conference on Machine Learning, pages 5887–5896, 2019.

### (2f)
Yaodong Yang, Jianye Hao, Ben Liao, Kun Shao, Guangyong Chen, Wulong Liu, and Hongyao Tang. "Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning". arXiv preprint arXiv:2002.03939, 2020.

### (2g)
- "Multi-Agent Actor-critic for mixed cooperative-competitive environments" by V. Lowe, Y. Wu, A. Tamar, J. Harb, I. Sutskever, and D. Silver (https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf):
This paper presents a multi-agent reinforcement learning algorithm that uses centralized training and decentralized execution, and demonstrates its effectiveness in a variety of multi-agent games.
Aim of counteract the problem of high variance with many agents of policy gradient methods by using an ensamble of policies.
MADDPG: Local actor use local info to deal with competition and cooperation (CTDE) while centralized critic has actions also about other agents. The action is communicated or got from inference of policy (communication of obs in this case)
?the inference is an alternative to communicate the action?
BAD: to handle different objective of agents (like competitors can get different reward structures for a same obs) the critic is per agent. To perform well in both coop/mixed env (be more robust and not overfit) use an ensamble of policies selecting which one to use to perform well and is costly.
The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) model presented by Lowe et al. (2017) also tries to achieve similar goals. However, they differ in the way of providing the coordination signal. In their case, there is no direct communication among agents (actors with different policy per agent), instead a different centralized critic per agent – which can access the actions of all the agents – provides the signal. Concurrently, a similar model using centralized critic and decentralized actors with additional counterfactual reward, COMA by Foerster et al. (2018) was proposed to tackle the challenge of multiagent credit assignment by letting agents know their individual contributions.

### (2h)
- **Counterfactual Multi-Agent Policy Gradients**
COMA uses a centralised critic to estimate the Q-function and decentralised actors to optimise the agents’ policies. In addition, to address the challenges of multi-agent credit assignment, it uses a counterfactual baseline that marginalises
out a single agent’s action, while keeping the other agents’ actions fixed. 
COMA learn a fully centralised state-action value function Qtot and then use it to guide the optimisation of decentralised policies in an actor-critic framework, an approach taken by counterfactual multi-agent
(COMA) policy gradients (Foerster et al., 2018), as well
as work by Gupta et al. (2017). However, this requires onpolicy learning, which can be sample-inefficient, and training the fully centralised critic becomes impractical when
there are more than a handful of agents.

### (3)
- **QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning** 
QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations. Extension to VDN since add obs e projection in wider hidden space
QMIX tries to address the challenges
mentioned above by learning factored value functions. By decomposing the joint value function
into factors that depend only on individual agents, QMIX can cope with large joint action spaces.
Furthermore, because such factors are combined in a way that respects a monotonicity constraint, each
agent can select its action based only on its own factor, enabling decentralised execution. However,
this decentralisation comes with a price, as the monotonicity constraint restricts QMIX to suboptimal
value approximations.
[Rashid et al., 2018] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning". In ICML, pages 4295–4304, 2018.
Sunehag et al. [46] propose Value Decomposition Networks (VDN), which learn the
joint-action Q-values by factoring them as the sum of each agent’s Q-values. QMIX [40] extends
VDN to allow the joint action Q-values to be a monotonic combination of each agent’s Q-Values
that can vary depending on the state
- theoretical part in MAVEN

### (3a)
- **MAVEN: Multi-Agent Variational Exploration** 
face MARL problem of inefficient exploration and says that through committed exploration we can solve that. "Maintain a policy and overtime that will lead to better rewards".
use a variable fixed per episode extracted from a latent space and a dopt a hierarchical approach. It has the aim to improve QMIX algorithm and escape suboptimal convergence in MARL. Centralized trainining with decentralized execution (CTDE def https://arxiv.org/pdf/1905.05408.pdf)

### (3b)
- **QPLEX: DUPLEX DUELING MULTI-AGENT Q-LEARNING**
QPLEX: mixer network decompose the objective in value and advantage network. Use attention as well. Usefull comment on QMIX
[Wang et al., 2021a] Jianhao Wang, Zhizhou Ren, Terry Liu, Yang Yu, and Chongjie Zhang. "QPLEX: Duplex dueling multi-agent Q-learning". In ICLR, 2021.


- **Local Advantage Networks for Cooperative Multi-Agent Reinforcement Learning**
Approach similar to QPLEX but instead use a dueling network factorized per agent. The value estimate is centralized while the advantage is local


                not used
- **DECENTRALIZED COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING WITH EXPLORATION**
This is hard. Introduce V-learning SGD, for cooperative situations in MARL. Use CTDE as always
-->

## centr, decentr, CTDE
The primary approach adopted in cooperative MARL consist in facing the environment as being centralized. Doing so, we can settle the non-stationarity problem benefiting from observing interactions between agents and observe the world with a global perspective. However that learning method can result in inflexible policies and may not generalize well to new situations, and the simplicistic assumption of centralization of the information can be hard to obtain, since usually such applications do not involve a central entity out of the test bed. A decentralized control approach, instead enable each agent in making its own decisions independently, without the need for a central controller or global coordination. The indipendent learnerning framework can obtain good empirical performance in several benchmarks [2a], anyway there are few theoretical guarantees for decentralized learning and the interpretability is often insufficient.
Many recent works focused on a hybrid of the two [2b,2c,2d,3,3b], where the global information are required only during the training phase freeing the algorithm during the test phase from the burden of continuos awerness of other agents behaviour. Centralized training and decentralized execution (CTDE) has been expanded following two main line of research which milstones are shared with standard MARL frameworks. Adopting an actor critic model Multi-Agent Deep Deterministic Policy Gradient (MADDPG) [2g] uses a centralized per-agent critic to estimate the Q-function and decentralised actors to optimise the agents’ policies. There is no explicit communication in there since the others' actions are obtained by inferring the respective policies. In a similar approach, COMA [2h] address the challenges of multi-agent credit assignment, common problem when using a centralized value function, using a counterfactual baseline that marginalises out a single agent’s action, while keeping the other agents’ actions fixed. Beside the important results obtiainable in this manner, AC models requires onpolicy learning, which can be sample-inefficient, especially when the state space is big.
On the other end, and this work is mostly aligned with this vision, a centralized Q-function [2d,3,2e;2f,3b], where the optimality is reached considering the relationship between joint action value and optimal local actions. [2d] propose Value Decomposition Networks (VDN), which learn the joint-action Q-values by factoring them as the sum of each agent’s Q-values. QMIX [3] extends VDN to allow the joint action Q-values to be a monotonic combination of each agent’s Q-Values that can vary depending on the state. Even if QMIX represented initially the state of the art in collaborative tasks like StarCraft II micromanagement tasks, it collected numerous critiques for the limitations in its representation ability due to the monotonic constraint. Numerous extension of this work were proposed, QTRAN (2e) learns an unrestricted joint action-value function and aims to solve a constrained optimisation problem in order to decentralise it; [3a] face the inefficent exploration that lead to suboptimality through committed exploration conditioning the policy with a variable, fixed per episode, extracted from a latent space; QPLEX [3b] takes advantage of the dueling network architecture to factor the joint Q function in a manner that does not restrict the representational capacity. In this thesis we will make use of the advance in this framework arguing that this limitation can be ignored by providing the factorized Q functions of enough information to act, leveraging the centralized Q function of some aspects of training optimization.

<!-- to verify if actually this is true and QMIX is ok, https://github.com/wjh720/QPLEX/blob/master/pymarl-master/src/modules/mixers/dmaq_general.py -->

<!--
### (4a)
- **Communication and interaction in a multi-agent system devised for transport brokering**
Brokering of structured messages from agents which want to realize their own goals. They claim for the necessity of a common language
"most physical robot teams do not use the agent communication languages developed by the multi-agent systems community. Instead, they typically use an ad hoc solution, defining their own protocols specific to their system. This approach lacks the semantic power and transparency afforded by a language like KQML and does not allow robots from various teams to communicate"
RQ: How should messages be generated, transmitted and represented? How can the content of messages be standardised? What principles (e.g. concepts, mechanisms and patterns) can be used?
BAD: they claim that to understand each other standardised and well structured messages are needed. They use a template to comunicate in their MAS environment.
Claim there is a need for a formal template (portability) and efficiency for fast messaging (not like ASCII messages or images)
RQ: Can we have both efficency and portability in robotic systems?
BAD: They do not learn the communication (not differentiable, just communicate when needed by the system) 
GOOD: They use a two tier communication (ACL main channel encapsulate a backchannel of fast communication tx)

### (4a17)
S. Poslad. Specifying protocols for multi-agent systems interaction. ACM Trans. Auton. Adapt. Syst., 2(4), Nov. 2007.

### (4a19)
F. Tim, R. Fritzson, D. McKay, and R. McEntire. Kqml as an agent communication language. In Proceedings of the third international conference on Information and knowledge management, CIKM, pages 456–463, 1994.

### (4b)
- **Communication Efficiency in Multi-Agent Systems**
Talk about ACL message format for communication and how these can be integrated in a robotic context in a way to have better efficency. Use a second channel of communication called backchannel. It is interesting for the discussion about the necessity of a structure in the message which can be contrapposta all'uso di un messaggio in the continue space.

### (4c)*
Fritz Heider and Marianne Simmel. "An Experimental Study of Apparent Behavior". The American Journal of Psychology, 57(2):243–259, 1944. ISSN 0002-9556. doi: 10.2307/1416950.

### (4d)*
Amir Rasouli, Iuliia Kotseruba, and John K. Tsotsos. "Agreeing to Cross: How Drivers and Pedestrians Communicate". arXiv:1702.03555 [cs], February 2017. arXiv: 1702.03555.


### (4e)
- **Learning Emergent Discrete Message Communication for Cooperative Reinforcement Learning**
They claim a discrete communication is better. Agent communication through a broadcast-and-listen mechanism. They use PPO CTDE since is cooperative, with attention mechanism when summing the messages (i.d. Mine with attention module in merging messages)
GOOD: They use a self attention module: multiplicative attention of Lowe 2015
BAD: the algorithm used has limitation of applicability but can be used with mod in a different context
They claim that discrete performance of communication is lower than continuos (and probably interpretability in messages should be dropped when agent communicatinrg between themselves) also they propose as extension the use of Vaswani 2017 module of dot product attention.
opposed to CommNet, IC3Net


### (4f) events
- **Event-Based Set-Membership Leader-Following Consensus of Networked Multi-Agent Systems Subject to Limited Communication Resources and Unknown-But-Bounded Noise**
Communication resources limited, Noise in process and measurements, all under a consensus paradigm with elected leader
RQ: How can one develop an effective consensus algorithm, which can provide a confidence region ensuring the inclusion of the true states of agents in the presence of external disturbance and measurement noise?
Treat it as an optimization problem (quite complex math reasoning)

### (5a) memory sharing
- **IMPROVING COORDINATION IN SMALL-SCALE MULTI-AGENT DEEP REINFORCEMENT LEARNING THROUGH MEMORY-DRIVEN COMMUNICATION**

DIAL and BiCNet  hidden

### (4g)
solve scalability issues related to variable number of agents by grouping in CTDE
- **Learning Communication for Cooperation in Dynamic Agent-Number Environment**
CTDE but to limit the explosion of complexity in training use divide the agents in groups. It doeas use a dot product attention in order to decide which agent will become the leader of a dynamic group. Himself choose who to include in the group. Has a communication channel working through LSTM, selecting the important parts (?). Use QMix to assign accordingly the rewards portions to every contributor
BAD: only applicable to cooperation

CommNet, IC3Net  obs

### (4h)
- **ACCNet: Actor-Coordinator-Critic Net for “Learning-to-Communicate” with Deep Multi-agent Reinforcement Learning**
Built on top of DIAL and BiCNet
WHAT: They propose a new model network (general framework) to ease the learning of communication protocols among agents. In the final architecture proposed they use two critics, where one is dedicated to the message network.
RQ: Can we learn multi agent communication protocols even from scratch under partially observable distributed environments with the help of Deep MARL?
BAD: act in PO but as well as CommNet is restricted to Fully cooperative. CTDE aka central coordinator represent a single point of failure. Does not present temporal consistency
GOOD: PO, MARL, no IL, AC, limited bandwidth of communication.

CommNet, DIAL, BiCNet, MADDPG and COMA are fully cooperative but MADDPG handle also competitive envs


### (5s)
- **Efficient Agent Communication in Multi-agent Systems**
address message passing and agent discovery in dynamic environments where agents arrive and go
BAD: Rely on an infrastructure of communication which in dynamic environments where there is not a center of communication and cannot exist a unique Actor Manager is not implementable

just reduce number of communications
### (4i)
- **Multi-Agent Graph-Attention Communication and Teaming**
Further of the interesting new attempt to give decide who should speak with who by adopting a module which produces a graph of attention.
### (4j)
- **Targeted multi-agent communication algorithm based on state control**
SCTC
RQ: reduce the communication channel trasmission without loose in results
BAD: They believe that the communication should happen only under some events and therefore require a customization ad hoc of parameters.
"can broadcast communication messages only when the current communication message changes greatly in comparison to the previously broadcasted communication message or aftern n steps"
Double attention mechanism self and external to acquire only interesting messages internally. Buffer of past messages maintained in order to reason at every step with comm of agents even when they send nothing (nothing new). Tested on StarCraft
### (5g)
- **When2com - Multi-Agent Perception via Communication Graph Grouping**
RQ: learning to construct communication groups and learning when to communicate in a bandwidth-limited way
Focus on the efficiency of communication (make groups no broadcast) by making direct handshaking with others and using a matching function to determine the correlation between information known by the agents (potentially to transfer) and an attention module, to decide when and with whom comunicate. In the specific it address the multiagent collaborative perception problem, where an agent is able to improve its perception by receiving information from other agents.  

### (4l) infer to avoid unuseful send
- **Learning Individually Inferred Communication for Multi-Agent Cooperation**
Learn with who you want to talk to reduce the overhead of communication in a collaborative setting. This is applicable to case where we have a critic centralized (join action-value), therefore it can be used also to reduce the communication in other appraoches like TarMac. They propose a two phase training in which the environment and network of action is trained first (or with whatever pre-trained CTDE algorithm) then the proposed netowrk of prior for communication is learnt after.
BAD: each agent has to know who is around him in order to decide who speak with


### (5a aside)
- **The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning**
RQ: Can an agent with self interest be damaging for a team of coordinated agents in a common environment with shared communication channel?
They show empirically how agents which ignore others rewards can assume adversarial behaviours changing properly ther own internal state to make the others do what they want.
BAD: it is true only when the others are freezed, if they are learning all together this is prevented.

### (5b)
- **Adversarial Attacks On Multi-Agent Communication**
Claim that we need to know the model of the attack in order to train our policy to prevent it. They do that for a gneral framework of attack by using having an agent acting the attacker which try yo get a surrogate perform wrost while training the surrogate to imitate the actor. Is just a training procedure framework in which we assume to know the other actor actions and improve itd policy.
RQ: 
GOD: the surrogate idea
BAD: homogeneous agents, works by broadcast and at all steps the agents exchange feature maps obtained from LIDAR views which are expensive.
we found that more practical transfer attacks are more challenging in this setting and require aligning the distributions of intermediate representations.

### (5a)
- **IMPROVING COORDINATION IN SMALL-SCALE MULTI-AGENT DEEP REINFORCEMENT LEARNING THROUGH MEMORY-DRIVEN COMMUNICATION**
BAD: This as well as many other approaches memory driven rely on a shared buffer of communication where they read and write to infer a shared representation. Does not scale well and is not quite apllicable to a dynamic environment with agents which arrive and go. Also, they use centralized critic
Memory driven approach with MADDPG (that is actor critic CTDE). The memory approach is used to reacha common representation of the environment between agents.
We introduce a shared communication mechanism enabling agents to establish a communication protocol through a memory device M of pre-determined capacity M (Figure 1). The device is designed to store a message m ∈ RM which progressively captures the collective knowledge of the agents as they interact. An agent’s policy becomes µθi : Oi × M 7→ Ai , i.e. it is dependent on the agent’s private observation as well as the collective memory. Before taking an action, each agent accesses the memory device to initially retrieve and interpret the message left by others. After reading the message, the agent performs a writing operation that updates the memory content. During training, these operations are learned without any a priori constraint on the nature of the messages other than the device’s size, M. During execution, the agents use the communication protocol that they have learned to read and write the memory over an entire episode. We aim to build a model trainable end-to-end only through reward signals, and use neural networks as function approximators for policies, and learnable gated functions as mechanisms to facilitate an agent’s interactions with the memory. The chosen parametrisations of these operations are presented and discussed below.

### (5aa)
Hu, Guangzhen et al. “Event-Triggered Communication Network With Limited-Bandwidth Constraint for Multi-Agent Reinforcement Learning.” IEEE transactions on neural networks and learning systems PP (2021): n. pag.

### (5d)
- **Multi-Agent Graph-Attention Communication and Teaming**
Further of the interesting new attempt to give decide who should speak with who by adopting a module which produces a graph of attention.

### (5g)
- **When2com - Multi-Agent Perception via Communication Graph Grouping**
RQ: learning to construct communication groups and learning when to communicate in a bandwidth-limited way
Focus on the efficiency of communication (make groups no broadcast) by making direct handshaking with others and using a matching function to determine the correlation between information known by the agents (potentially to transfer) and an attention module, to decide when and with whom comunicate. In the specific it address the multiagent collaborative perception problem, where an agent is able to improve its perception by receiving information from other agents.  


### (5h)
- **Learning Efficient Multi-agent Communication: An Information Bottleneck Approach**
They treat the communication badwidth problem faced with a scheduler which should deliver a lot of messages probably redudant or unecessary therefore on the side of the scheduler an additional processing operation is done here to limit the unnecessary communications.
GOOD: to address the similarity of messages (and so be able to merge them) they make sure the messages respect an upperbound in the entropy
-->


- structured unstructured
- learned or not
: my claim
- discrete or continue
: my claim
- methods of transmission
: my claim
- efficency directed, save messages [1b]
- graph and teaming + attention
: my claim
- focus on cooperation

# who is continuos (6a, 6c, 6f)
# who is discrete
# who broadcast message (6a in a transformer to use attenion, 6b not use att, 6c use centralized controller which can be considered as att filter in broadcast behaviour)
# who send directed messages
# who use only cooperation (6c)
# [4l]v is efficiency of communication related "inference to avoid unuseful send"
# [4h]v or ACCNet, is "learning to communicate" but place himself in comparison of other knonw methods (CommNet,DIAL,BiCNet,MADDP,COMA)
# [4i]v in terms of graph, attention, teaming can be used also elsewhere 
# [4g]x solve scalability issues related to variable number of agents in CTDE

The gain in autonomy given by decentralization and limited visibility of agents to reduce their state space, comes at the expense of a greater degree of uncertainty and variability in the environment exploration. Even if at training time each agent can access to complete knowledge, this can be not enough to learn an optimal behavior, resulting in suboptimal trained agents and inefficient interactions at test time especially when complex coordination is required. Therefore mechanism of information sharing and communication are introduced to reduce the non stationarity effects [4c,4d].
Communication between agents can be either explicit using cheap-talk channels, or implicit, by observing other agents’ actions or their effect on the environment. in case of explicit communication a preferrable choiche can be to rely on standard format of messages, Agent Communication Language (ACL), in order for indipendent agents to be able to converse with a precisely defined syntax, semantics and pragmatics [4a17,4a19]. On the other hand, ad hoc communication protocols with learnt communication are mostly being adopted when complex coordination is required. Even if the first can result in a detriment of generality and flexibility given the imposed structure having a standardised and well defined structure is helpful when the intent is the common understanding, indeed these properties are interesting in general and some work of research try to take the best of both aspects [4a,4b]. Even if this thesis work focus on a differentiable version of the communication do not neglect a sort of encapsulation provided by underneath protocols or enrichment of the communication syntax still maintaining the benefits of a learnt communication language.
Efficiency is key of proposed implementations to be effectively transposed in real world environments where dynamism, security, and message brokering are not just details the environment [4b,5s,5h], but is often hard to try to design a solution that fits all.
In particolar there are conflicting views about the format of the message (structured, discrete, continuos, ...)[4b,4e,6c,2g] and the optimal method of information exchange in terms of cost benefits. [5a] use a common memory buffer, where agents can write and read to share information, [5aa] adopt a en event based framework where communication happen only under certain circumstances, while [4l] adopting an implicit communication show that simple coordination can be obtained just observing the others actions and inferring their policies. More commonly a direct message is shared between two agent by opening a communication channel [4j,4i] or by broadcasting it to everyone [6a,6b,6c,6e]. Obviusly, this last appraoch, is the most expressive nder partial observability assumptions, but it is even the most expensive in terms of transmission traffic and present the drawback that in crowded spaces the useless noisy information can easily overwhelm the relevant information.
[4g,5d,5g,4i,6g,6h] aim at the creation of smaller groups of agents focusing on inter and intra correlation to limit irrelevant reasoning and better coordination performance. [5g,4l] reduce the messages sent by learning when the communication is truly necessary, or [4l] when information are redundant and avoidable. [4j] solve the problem by targeting the communication towards who is interested, [6e,4e,6a,6c] instead act on the side of the listener by adopting different mechanisms of attention in order to filter out the messages of irrelevant agents for the current action choice. When using attention or similar concepts, the message being processed is typically filtered or weighted based on its relevance for the agent itself, ignoring its relevance in relation to other messages. This in addition to the lack of structure to frame in time the messages, reduce the ability of coordination between multiple agents and maintain consistency in their decisions over time. Lastly, the design of the architecture is often crafted with the goal of fostering cooperation [2h,4e,4g,4h,4l,6c]. However, it can be more challenging to design for interactions in mixed environments. There with no strict assumption of collaboration is is important to also consider the possibility that agents may act with self-interest or with adversarial intentions, as analyzed in [5b,5aside].

<!--
Many works therefore resort to differentiable communication [7, 18, 30, 34, 43], where agents are allowed to directly optimize each other’s communication policies through gradients. Among them, Choi el al. [7] explore a high-level idea where the communication is implemented by generating messages that the model itself can interpret.

Learned communication can improve the efficiency of agents in interacting without having to rely on a hand crafted communication channel. As drawback it's difficult to learn a communication language as it typically require multiple timesteps to execute, and this can limit the agility of control during execution (Tian et al., 2020). By contrast, coordination based on common knowledge is simultaneous, that is, does not require learning communication protocols (Halpern and Moses, 2000).
-->


<!--
### (4f) not used
- **Event-Based Set-Membership Leader-Following Consensus of Networked Multi-Agent Systems Subject to Limited Communication Resources and Unknown-But-Bounded Noise**


### (6a)
- **COMMUNICATION IN MULTI-AGENT REINFORCEMENT LEARNING: INTENTION SHARING**
How to harness the benefit of communication beyond sharing partial observation (as in TarMac or ATOC)
They propose a new intention schema for sharing info describing the intentions of the agents as future trajectories.
Claim: messages encoded of current observation or a hidden state of LSTM, under partially observable environments, are useful but do not capture any future information". critic to Ic3Net and CommNet
"sharing intention is more important than having a prediction of the future" as a traejctory they use the pair observation plus action but they consider H-steps in the future where only the first is real and the next imagined. Works autoregressively in the computation of trajectory at every step
BAD: the policy takes the message m and the observation but in autoregressive step they use the current t-1 with future imaginated observation to obtain the next action, also rely on predicted opponent actions on the base of own observation: indeed in partial observeability the attention is all on first and second step (i.e. the trajectory is mostly useless) while with full observability the trajectory is usefull
GOOD: they predict the change from obs to obs to reduce bias, The trajectory is processed by an attention module (scaled dot product)
Partial observability rende many immaginated steps unusefull
<!--
- predictor delle azioni di tutti meno la tua dalla tua observation
- tua obs+f(tua obs+tua azione+tutte le altre azioni)=next observation
- con la next obs e il messaggio prev ottieni la next action
- e ricominci il ciclo
- tutte le coppie [(obs, action), ...] vengono passate nell'attention module
-- >
Interessante loro modello di attenzione e trajectory immagination but (Propongono come estensione) can be improved using it in better framework of message passing and info aggregation (cioè hai le intentions ora le devi usare per prendere buone decisioni)

### (6b)
- **Learning to Ground Multi-Agent Communication with Autoencoders**
RQ: Should we first learn a common language of communication before expecting to communicqte with success
"The main challenge of learning to communicate in fully decentralized MARL settings is that there is no grounded information to which agents can associate their symbolic utterances. This lack of grounding creates a dissonance across agents and poses a difficult exploration problem. Ultimately, the gradient signals received by agents are therefore largely inconsistent."
<!--ehm boh, mi sembra molto strano...-- >
BAD: they have a shared module for processing the communication messages which are all concatenated together. No attention. Also processed communication messages concatenated with image features encoded are passed through a GRU module to extract the policy action.
Also they use an autoencoder local for everyone to encode and decode the image from which extract the communication message.

### (6c)
- **Learning Multiagent Communication with Backpropagation**
CommNet from 2016 is a first attempt of including a learned communication channel: there is a single controller which output actions for every agent and take in conmsideration all agent's states. It averages the hidden states for centralized communication
Is usable only in environmnet cooperative.
BAD: no structure just take all together and spit output  htat can be doable in very easy MARL but when mthere are complex relations it become not scalable. Suffer of the credit assignment problem for multi agents.


### (6d)
- **LEARNING WHEN TO COMMUNICATE AT SCALE IN MULTIAGENT COOPERATIVE AND COMPETITIVE TASKS**
IC3Net as extension of CommNet as well use a continuos vector
Averages the hidden states for centralized communication
Learn also to communicate only with some of the all individuals. Can be extended to mixed but not scale well since the conflict of interests with multiple target 
//BAD: They require a central aggregator for messages or a mechanism of sync to decide the action accordingly to all surrounding actors.//

### (6e)
- **TarMAC: Targeted Multi-Agent Communication**
TarMAC introduce targeted communication to let coordination arise. The main part is a module of Multi head attention used to decide how much of the information coming from other to retain and consider. Both sender and receiver have to produce a similar signature in order to have higher consideration. This give a lot of responsabilities of choice to the receiver which unconscious of the info coming should decide to hear. (mine has ever the choice on the listener, more specifically on it's coordinator, but this is trained separately to the policy used to move therefore can choose with more liberty)  
It state also about the necessity of continuos vector as said by CommNet and ICNet, and centralized training since coordination is not enough in complex (like ad) scenarios.
Also we have to say that they used an approach of policy shared weights in their experiments therefore we have no guarantee that slightly different policies will fit in the behaviour of the group. AC method. The centralized critic have actions and hiddens.

### (6f)
- **Multiagent Bidirectionally-Coordinated Nets**
BiCNet / ATOC
Both use a bidirectional recurrent network as a communication channel. They fix the positions of agents in the bidirectional recurrent network to specify their roles.
CommNet is a single network designed for all agents. The input is the concatenatio of current states from all agents. The communication channels are embedded between network layers. Each agent sends its hidden state as communication message to the current layer channel. The averaged message from other agents then is sent to the next layer of a specific agent. However, single network with a communication channel at each layer is not easy to scale up.
More recent approaches for centralised learning require even more communication during execution: CommNet (Sukhbaatar et al., 2016) uses a centralised network architecture to exchange information between agents. BicNet (Peng et al., 2017) uses bidirectional RNNs to exchange information between agents in an actor-critic setting. This approach additionally requires estimating individual agent rewards.

### (6g)
- **Learning Attentional Communication for Multi-Agent Cooperation**
ATOC attentional communication model to learn effective and efficient communication under decpomdp environment for large-scale MARL. An initiator agent has the ability to form a local group for the exchange of encoded observation for coordinated strategies. AC model with bidirectional LSTMs as communication channel. All with weights shared therefore this is suitable to large scale environments. Differently from BiCNet, use encoded observations and attentional communication model in a way to send the relevant information.
Exploit a bidirectional LSTM unit as the communication channel to connect each agent within a communication
group. The LSTM unit takes as input internal states (i.e., encoding of local observation and action
intention) and returns thoughts that guide agents for coordinated strategies.
Does similar as me but on the side of the sender not of the receiver


- **Multi-Agent Concentrative Coordination with Decentralized Task Representation**
non è commnication
Coordination for reaching subtasks by enhancing the centralized q mixing network with environment information specific of the subtaks
BAD: not really partial observability

### (7a)
impose a goal consistency between the objective selected by the agents in a cooperative task but is not communication. They use a goal comune learned to reduce useless exploration
- **Goal Consistency: An Effective Multi-Agent Cooperative Method for Multistage Tasks**
They have a set of predefined goals. A general observation value relative to environment and a vector of observations goals related. An attention module gives weights to select at each time which goal is the one that in relation with the current general observation is more important for the action selection.
BAD: being a CTDE the critic works with the whole vector of obs from all agents + all actions; there is no message passing during execution so this does not generalize, also they have all same goals and intentions so they should only coordinate on the order.
They use losses between probability distributions of different agents for consistency...
GOOD: having two soft attention modules (local and global/centralized) you have two probabilities distributions, you can use the second to influence the first

### (7b)
perfect example of Learned Hierarchical Consensus
- **Structured Diversification Emergence via Reinforced Organization Control and Hierarchical Consensus Learning**
RQ: how do the agents form teams spontaneously, and the team composition can dynamically change to adapt to the external environment? How to maintain diversity in the behavior of different teams and to form tight cooperation between the agents within the team?
Merges some learned parts with coded wanted constraints
Graph theory algorithm in first organizational module (not differentiable) to make adaptive teaming decisions. The hierarchical consensus module then generate the team intention and the individual intention based on the obtained teaming results. Finally, the decision module outputs the structured diversification policies
preformation of groups before reasoning over intention for policy?
they use the trick of global state during training to have better results...

### (7c)
Ren, Wei and Randal W. Beard. “Consensus seeking in multiagent systems under dynamically changing interaction topologies.” IEEE Transactions on Automatic Control 50 (2005): 655-661.

### (7d)
Olfati-Saber, Reza et al. “Consensus and Cooperation in Networked Multi-Agent Systems.” Proceedings of the IEEE 95 (2007): 215-233.
-->

# send hiddens (6c, 6d, 6e, 6f)
# merge wheightened hiddens (6c, 6d)
# use crafed messages ad hoc (6a but not per agent, 6e head outputting, 6g)
# of them send specialized per agent messages
# use wheight sharing to have same reasoning ability (6a, 6e, 6f, 6g)


## What, when communicate and how to include the info in your choices (implementation architectures of such methods e.g. he have done this in this way)
The content of the message is nonetheless the most important detail for the recipient agent and when designing it a trade-off between expressivness and generalizability should be found. The message itself should have the aim of provide additional information regarding the point of view of the communicator in a way to ease the resoning of all listener reducing uncertainty about his behaviour and so allow coordination. One option could of using a very expressive message which encapsulate the reasoning process of the agent itself: [6c,6d,6e,6f] structured their architectures around recurrent modules, using the hidden state as signal message for others. Usually this encode past and current information, which can be not so helpfull if not misleading for receipient, but it can be properly adapted in the learning process to be expressive also for who receive it. However merging the two reasoning operations in a same step could be limitating in difficult tasks. [6c,6d] in addition merge the multiple incoming messages into a single communication vector by using weighted operastions which do not help a strong coordination when many agents are present during the learning process.
[6e,6g] use crafted messages to transmit elaborated information to others introducing the necessity for everyone to be able to understand in first place the message or forcing the architecture under a weight sharing framework to have same reasoning ability on a piece of information. Some research works in particular [6a,6e,6f,6g] use this approach in their policy also to counteract scalability issues.
In a simpler transmission approach [5g,6a] use the current or time delayed observation as communication messages, usually with additional information expressing intention. Such implementation delegate the most of the work of interpretation and coordination to the receiver but allowing also more flexible dynamics of interaction. Nevertheless a common critic in this case is due to the expensive messages trasmitted since the information shared are raw, but a solution can be found in learning a better representation for the observation without loosing too much information. For instance [6b] analyzing the problem of enstablishing a common language between agents propose an autoencoder which intermediary representation is effectively used as message.
Finally, great effort is dedicated to the effective learning of the policy which should act in coordination with other entities following individual observation and incoming messages. Previous works, [7c,7d], impose a necessary consensus to be able to preoceed therefore forcing the agreement before acting, while nowdays in learned framework a looser agreement is to be preferred, aiming instead to obtain convergence in the target choice and coordination in actions: [7b] doing so form groups of agents with similar objectives to have tighter cooperation and diversity of action between teams, but the proposed architecture is not end-to-end differentiable. On the other hand [6f,6g] process incoming information together with the internal state in RNNs module to obtain a coordinated policy.
The main point of this work focus on the belief that instead the coordination should be considered in general, only circumstantiated to some situations and not being predominant at every step, since extensive reasoning on others beliefs and intention can slow the convergence to a good policy till the point to be even harmful. 

<!--
Memory mechanism A common way to tackle partial observability is the usage of deep recurrent neural networks, which equip agents with a memory mechanism to store information that can be relevant in the future (Hausknecht and Stone 2015). However, long-term dependencies render the decision-making difficult since experiences that were observed in the further past may have been forgotten (Hochreiter and Schmidhuber 1997). Approaches involving recurrent neural networks to deal with partial observability can be realized with value-based approaches (Omidshafei et al. 2017) or actor-critic methods (Dibangoye and Bufet 2018; Foerster et al. 2018b; Gupta et al. 2017). Foerster et al. (2019) used a Bayesian method to tackle partial observability in cooperative settings. They used all publicly available features of the environment and agents to determine a public belief over the agents’ internal states. A severe concern in MADRL is that the memorization of past information is exacerbated by the number of agents involved during the learning process.

Bidirectional RNN allow to maintain the internal state of the agent e share info with collaborators however, it assumes that agents can know the global Markov states of the environment, which is not so realistic except for some game environments. For this reason the BiRNN network used is not to decide an action but to decide the coordination: if the state of an agent is relevant for another and this should take in consideration his information, the coordiknator will decide to include his message between the information used by the policy to take a decison in the next action 
-->

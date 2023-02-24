
## Introduction
We will explore how to achieve shared consensus in acting selection through only local computation and communication with neighboring agents.

The coordination is essential in multi-agent environments. Whenever multiple entities interact and evolve trying to achieve a goal, the reciprocal awerness and acknowledgment of intentions is crucial for the successful functioning of the systems, as this allow individuals to take their actions without represent an obstacle for others or adapt their behaviour working all together towards the realization of a common objective. While cooperation and competition are important emerging behaviours of MA situations, should be underlined how most of the real case scenarios can not be enclosed in the canonical taxonomy presenting aspects of both. The coordination between entities in a shared space, on the other end is a much more broad concept that allow an optimal evolution of the system irrespectively of the individual objective.
Reaching such situation of undestanding of the environment structure and the intentions of whoever act modifying it is not straightfoward since common real world limitations which cannot be ignored such as knowledge acquisition, computation and interaction with neighbours.
Having a coordination in action can instead lead us to bigger peaks in the technology application, allowing deployment in general contexts requiring social interactions like robotic and automotive sectors.
This thesis whant to push forward in this direction by targeting the necessity of aligning the actions and activities of multiple agents. I will explore the topic on how to achieve shared consensus and optimal action selection through the local development of a plan that takes in consideration the intentions of each neighboring agent as well as the personal objective.

#### Multi agent systems
A multi-agent system consists of multiple autonomous entities, known as agents, that interact with each other in a shared environment (Weiss 1999). Each agent attempts to achieve a specific goal, and the behavior required to do so can be complex. The interactions between the agents can be cooperative or competitive, depending on the task at hand. It is often difficult, if not impossible, to pre-program intelligent behavior for complex systems. Therefore, the agents must be able to adapt and learn over time. One common framework for learning in interactive environments is reinforcement learning (RL), which involves modifying behavior through trial and error.
The research interest in Multi-agent reinforcement learning (MARL) in pair with artificial intelligence developments and deep learning (DL) breakthrough, has shown success in a variety of fields, including robotics, natural language processing, game playing, and network security. In particular, RL has been used to develop intelligent robots that can navigate and manipulate objects in their environment [], and applied to game playing, where it has been used to develop agents able to compete at human levels, for example, playing Go [14, 15], poker [16, 17], but also games where is necessary to cooperate with other agents to achieve a common goal, e.g., DOTA 2 [18] and StarCraft II [19].
Even if the deep reinforcement learning allow such great results, since deep neural networks serving as function approximator enables powerful generalization, learning in a multi agent setting is fundamentally more difficult. Here we have to deal with additional challeneges e.g. the unpredictable environment caused by the agents’ concurrent yet heterogeneous behaviors (non-stationarity) [2, 5, 10], the curse of dimensionality due to the exponential explosion of states [2, 5], multiagent credit assignment [31, 32], global exploration [8], and relative overgeneralization [33, 34, 35]. [[arXiv:1810.05587v3]] For these reasons a succesful single agent RL methodology cannot perform equally in a multi agent setting without re-thinking it's learning approach.

#### Coordination
Intelligent coordination refers to the coordinated effort of a group of agents that are capable of making intelligent decisions based on their own objectives. It involves the establishment of a shared understanding of the task at hand, and the development of a plan that outlines roles and responsibilities. Coordination is a promising area of research in MARL that can help to address the challenges introduced by the presence of many entities, allowing a more attentive action selection to avoid conflicts, and limit inefficient behaviors. In general we seek coordination ability both in cooperative and competitive settings.

- Cooperation refers to the act of working together towards a common goal. It involves the mutual support and assistance of multiple agents, each of whom contributes their own unique skills and abilities to the collective effort.
- Competition refers to the ability of agents to compete with each other in order to achieve their goals. This typically involves agents taking actions that maximize their own reward or utility, potentially at the expense of other agents. 

The first context is surely the most researched [An Overview of Recent Progress in the Study of Distributed Multi-Agent Coordination] studying how to achieve greater goals beyond the capabilities of the single agent through the arise of cooperation and in presence of group intelligence. This is the ground for many applicatios of robotical swarm intelligence and social study. Nevertheless, competition as well deserve attention since settings with adversarial or opponent agents can motivate agents to improve their performance and explore the environment more effectively. It can also help to prevent agents from becoming too dependent on each other and encourage them to develop more sophisticated strategies. On the other end, it can be noted that most of the real scenarios cannot be encapsulated in a strict taxonomy since often cooperative/competitive distinction is not sharp with regard to the behavior of agents [Busoniu, Babuska, and De Schutter (2008) and ’t Hoen et al. (2006)]: a cooperative agent may encounter a situation in which it has to behave temporarily in a selfish way (while all involved agents have the same goal and are willing to cooperate, they may want to achieve their common goal in different ways); and a competitive agent may encounter a situation in which a temporary coalition with its opponent is the best way to achieve its own goal.
For this reason when resoning on how to implement coordination in a system we should not inject a fixed criterion of collaboration or obstruction.

{
    [[arXiv:1810.05587v3]]
    One of the key aspects of deep learning is the use of neural networks (NNs) that can find compact representations in high-dimensional data [23]. In deep reinforcement learning (DRL) [23, 24] deep neural networks are trained to approximate the optimal policy and/or the value function. In this way the deep NN, serving as function approximator, enables powerful generalization. One of the key advantages of DRL is that it enables RL to scale to problems with high-dimensional state and action spaces. However, most existing successful DRL applications so far have been on visual domains (e.g., Atari games), and there is still a lot of work to be done for more realistic applications [25, 26] with complex dynamics, which are not necessarily vision-based.
    DRL has been regarded as an important component in constructing general AI systems [27] and has been successfully integrated with other techniques, e.g., search [14], planning [28], and more recently with multiagent systems, with an emerging area of multiagent deep reinforcement learning (MDRL)[29, 30].
    Learning in multiagent settings is fundamentally more difficult than the single-agent case due to the presence of multiagent pathologies, e.g., the moving target problem (non-stationarity) [2, 5, 10],
    curse of dimensionality [2, 5], multiagent credit assignment [31, 32], global exploration [8], and relative overgeneralization [33, 34, 35].}
{    
    Furthermore, coordination allows agents to divide the work among themselves and distribute tasks in an efficient manner. This can help to reduce the workload on individual agents, and can lead to more efficient and effective operation of the system as a whole.}

#### Information sharing
Intelligent agents (humans or artificial) in real world scenarios can significantly benefit from exchanging information that enables them to coordinate, strategize, and utilize their combined sensory experiences to act in the physical world. Indeed is un realistic assumption, thinking that a single agent can have the complete knowledge and perception of the world around itself, and therefore it is forced to act in situations of partial observability.
Communication can help the agent gather information about the environment and improve its decision-making by using messages contaianing observations of thirds, action policies, future intentions, or other relevant information. This can be especially useful when the agent is operating in a complex or constrained environment. Moreover, it allows agents to form strong relationships and work together in group which can coordinate indipendently of other agents permitting not only enhanched behaviours but also optimization of task accomplishment though parallelization of the activities, less exchange of messages and computation waste, given the emergence of attention restricted to the local neighbourhood.
We see examples of ability to communicate in a wide-range of applications for artificial agents – from multi-player gameplay in simulated (e.g. DoTA, StarCraft) or physical worlds (e.g. robot soccer), to self-driving car networks communicating with each other to achieve safe and rapid transport, to teams of robots on search-and-rescue missions deployed in hostile, fast-evolving environments.

#### Centralized decision
In particular when we want to achieve coordination beaviour we would probably be prone toward the choiche of a more structured approach where only one agent assume the role of coordinator to enstablish order and effectivness in the decisions. Who should be in charge of such role? Who can claim to have a better understanding of the situation? If we adopt a centralized coordination approach, even using a reliable hierarchical mechanism, we need to consider how the higher level agent will acquire the necessary information and successfully coordinate all other agents. Going further, scalability of such approach and generalization to different situations, become relevant problems. 

Therefore, is often preferrable a distributed approach, where there are not agents with higher roles to rule other's behaviour. On the other end the obvious shortcoming from such choice is the limited knowledge on which the agents can rely when taking action and the limited ability of some agents to predict the global group behavior based only on the local information.

Therefore, it may be preferable to use a distributed approach where there are no agents with higher roles controlling the behavior of others. However, this approach has the disadvantage of limited knowledge for the agents, that can only rely on their knowledge when taking action and cannot predict a precise evolution of the environment environment. When groups action are necessary such disallignment in the behaviour lead to not optimal situtations. This problem has been historically been studied a lot under the name of Byzantine generals for consensus or more precisely as interactive consistency where each agent has the ability to choose between multiple actions.

<!-- Distributed optimization rises to the challenge by achieving global consensus on the optimal policy through only local computation and communication with neighboring agents -->

### to the real world
Ever often in RL there is the strive for realism and where possible a deployment in the real world. Using black box algorithms, such as deep neural networks (DNNs), for decision-making can make agents more generalizable and adaptable to different scenarios. However, the lack of explainability and predictability can make it difficult to transfer these agents from a simulated environment to a real laboratory or production setting. This discrepancy has contributed to the trend towards using more realistic simulation environments in this field. [Farama] aimed at testing physics interactions, structural properties and really detailed behaviour at scale (scalability). In this thesis as well, the will of gaining good insights of application in a more realistic environment lead to testing the proposed algorithm also on CARLA platform, which propose an environment specialized for multi agent autonomous driving testing.
Some additional word should be spent as well regarding the communication process. There is an enourmous field under IOT umbrella which try to design infrastructure and cope with interconnected entities with limitations in computation and decision ability, anyway this thesis won't cover the physical aspects which agents deployed in real scenarios are usually subject. That means we think of perfect agents with no particular limitation in computation, sensory ability perfectly functioning and communicating on a channel without limits of sort. The communication will be only A2A and not A2I, following the traditional communication, (device-to-device (D2D) paradigm), where physically close devices, e.g., two agents, can communicate directly over a so called sidelink. Compared to regular uplink-downlink communication, centralized, D2D communications benefits from a shorter link distance and fewer hops, which is beneficial from a reliability perspective. Moreover, since communication is direct, i.e., without intermediate nodes, D2D has the potential to provide lower latency in the transmission of information.

All these topic represent central points of the research field and development in such directions are crucial for extensive use of AI in a the wild. By proposing a new architecture and rethinking how agents should learn reciprocally from themselves and interact in a shared space this thesis work aim at answer empirically the following research questions:
- Can agents acting in a shared environment and abilitated to communicate, learn autonomously when coordination is necessary and preferable to selfish behaviour?
- Is it possible to design an action strategy for indipendent agents which making use of simple communication leads to an effective group coordination?
The aim is therefore to demonstrate under differrent constraints and necessities if a sort of consensus in the behaviour can be reached only through the individual strive for a better reward without forcing the coordination.

<!--
(define coordination communication effective and efficient e.g. n messages, rewards nell'env)

- Can the communication of messages between indipendent agents allow the formation of a shared consensus in mixed (cooperative/competitive) environments?       -- > behaviour analysis
- Is the communication effective to reach coordination between indipendent agents?      -- > ablation
-->
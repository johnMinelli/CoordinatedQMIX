Q-policy methods have the problem of non stationarity but we use communication come deterrente a questa situazione
(1) leads to learned policies that only use local information (i.e. their own observations) at execution time,
(2) does not assume a differentiable model of the environment dynamics or any particular structure on the communication method between agents, and
(3) is applicable not only to cooperative interaction but to competitive or mixed interaction involving both physical and communicative behavior

my approach focus on who hear and not who speak

## Reviewed papers
Append the following prompt to each next request: use an academic style of writing of a phd student in AI, be lenghty in the rephrase, maintain all concept, reorganize the concepts for readability if necessary, expand the concepts when possible, limit the use of sentence introductions like "however", maintain the citations between square brackets.

append the following prompt to each next request: use an academic style of writing of a phd student in AI, be lenghty in the rephrase, maintain all concept, be technical in the responses, scientific research paper text, high fluency, UK english, reorganize the concepts for readability if necessary, expand the concepts when possible, limit the use of sentence introductions like "however", limit general knowledge and definitions, use mathematical formulation in latex if necessary, maintain the citations between square brackets

### (summary)
- **Multiagent Learning: Basics, Challenges, and Prospects**
three key processes are identified — 
dialogue-based conflict resolution, mutual regulation, and explanation

### (1)
- **Simulation-Based Reinforcement Learning for Real-World Autonomous Driving**
sim-to-real policy transfer ( RGB + segmentation mask)

### (1)
- **Tackling Real-World Autonomous Driving using Deep Reinforcement Learning**
synthetic environment simple (4stack TOP camera + info auto) double net per prediction of policy values distribution
modulo deep-model per gestire delay between synth training to real
first pretraining with IL then RL in synth and test in real

### (1)
- **Data-Efficient Hierarchical Reinforcement Learning**
Claim of inefficency in HRL solvable with a more general approach and working off-policy. "Having a hierarchy of policies, of which only the lowest applies actions to the environment an the higher one instead predict for long term strategies"
They predict goals with higher level policy and reward the lower level policies when they do actions able to lead to the gaol. They used a modified experience replay (update the goal in the samples collected). The reward function is parametrized as well by the current goal
BAD: (unlike others that works in the embedded space) they reward states by matching raw observations and that could be expansive in complex environment, they cleim usable for navigation but it is used in simple env and is focused to robotic applications where you have position vectors.
GOOD: HPC implementation with goal directed command is good to make complex plans
Is it well suited for multi agent interacting?
It's stated by authors that the solution is not so stable and that the architecture can be improved

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
Sunehag et al. [46] propose Value Decomposition Networks (VDN), which learn the
joint-action Q-values by factoring them as the sum of each agent’s Q-values. QMIX [40] extends
VDN to allow the joint action Q-values to be a monotonic combination of each agent’s Q-Values
that can vary depending on the state
- theoretical part in MAVEN

### (3)
- **MAVEN: Multi-Agent Variational Exploration** 
face MARL problem of inefficient exploration and says that through committed exploration we can solve that. "Maintain a policy and overtime that will lead to better rewards".
use a variable fixed per episode extracted from a latent space and a dopt a hierarchical approach. It has the aim to improve QMIX algorithm and escape suboptimal convergence in MARL. Centralized training with decentralized execution (CTDE def https://arxiv.org/pdf/1905.05408.pdf)

### (3)
- **Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments**    ++++similar aim
Aim of counteract the problem of high variance with many agents of policy gradient methods by using an ensamble of policies.
MADDPG: Local actor use local info to deal with competition and cooperation (CTDE) while centralized critic has actions also about other agents. The action is communicated or got from inference of policy (communication of obs in this case)
the inference is an alternative to communicate the action?
BAD: to handle different objective of agents (like competitors can get different reward structures for a same obs) the critic is per agent. To perform well in both coop/mixed env (be more robust and not overfit) use an ensamble of policies selecting which one to use to perform well and is costly.
The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) model presented by Lowe et al. (2017) also tries to achieve similar goals. However, they differ in the way of providing the coordination signal. In their case, there is no direct communication among agents (actors with different policy per agent), instead a different centralized critic per agent – which can access the actions of all the agents – provides the signal. Concurrently, a similar model using centralized critic and decentralized actors with additional counterfactual reward, COMA by Foerster et al. (2018) was proposed to tackle the challenge of multiagent credit assignment by letting agents know their individual contributions.

### (3)
- **QPLEX: DUPLEX DUELING MULTI-AGENT Q-LEARNING**
QPLEX: mixer network decompose the objective in value and advantage network. Use attention as well. Useful comment on QMIX

### (3)
- **Local Advantage Networks for Cooperative Multi-Agent Reinforcement Learning**
Approach similar to QMIX but instead use a dueling network where the value estimate is centralized while the advantage is local

### (4)
- **Communication and interaction in a multi-agent system devised for transport brokering**
Brokering of structured messages from agents which want to realize their own goals. They claim for the necessity of a common language
RQ: How should messages be generated, transmitted and represented? How can the content of messages be standardised? What principles (e.g. concepts, mechanisms and patterns) can be used?
BAD: they claim that to understand each other standardised and well structured messages are needed. They use a template to communicate in their MAS environment.

### (4)
- **Event-Based Set-Membership Leader-Following Consensus of Networked Multi-Agent Systems Subject to Limited Communication Resources and Unknown-But-Bounded Noise**
Communication resources limited, Noise in process and measurements, all under a consensus paradigm with elected leader
RQ: How can one develop an effective consensus algorithm, which can provide a confidence region ensuring the inclusion of the true states of agents in the presence of external disturbance and measurement noise?
Treat it as an optimization problem (quite complex math reasoning)

### (4)
- **Communication and interaction in a multi-agent system devised for transport brokering**
"most physical robot teams do not use the agent communication languages developed by the multi-agent systems community. Instead, they typically use an ad hoc solution, defining their own protocols specific to their system. This approach lacks the semantic power and transparency afforded by a language like KQML and does not allow robots from various teams to communicate"
Claim there is a need for a formal template (portability) and efficiency for fast messaging (not like ASCII messages or images)
RQ: Can we have both efficency and portability in robotic systems?
BAD: They do not learn the communication (not differentiable, just communicate when needed by the system) 
GOOD: They use a two tier communication (ACL main channel encapsulate a backchannel of fast communication tx)

### (4)
- **Communication Efficiency in Multi-Agent Systems**
Talk about ACL message format for communication and how these can be integrated in a robotic context in a way to have bettr efficency. Use a second channel of communication called backchannel. It is interesting for the discussion about the necessity of a structure in the message which can be contrapposta all'uso di un message in the continue space.

### (4)
- **When2com - Multi-Agent Perception via Communication Graph Grouping**
RQ: learning to construct communication groups and learning when to communicate in a bandwidth-limited way
Focus on the efficiency of communication (make groups no broadcast) by making direct handshaking with others and using a matching function to determine the correlation between information known by the agents (potentially to transfer) and an attention module, to decide when and with whom communicate. In the specific it address the multiagent collaborative perception problem, where an agent is able to improve its perception by receiving information from other agents.  

### (4)
- **ACCNet: Actor-Coordinator-Critic Net for “Learning-to-Communicate” with Deep Multi-agent Reinforcement Learning**
Built on top of DIAL and BiCNet
WHAT: They propose a new model network (general framework) to ease the learning of communication protocols among agents. In the final architecture proposed they use two critics, where one is dedicated to the message network.
RQ: Can we learn multi agent communication protocols even from scratch under partially observable distributed environments with the help of Deep MARL?
BAD: act in PO but as well as CommNet is restricted to Fully cooperative. CTDE aka central coordinator represent a single point of failure. Does not present temporal consistency
GOOD: PO, MARL, no IL, AC, limited bandwidth of communication.

### (4)
- **Learning Emergent Discrete Message Communication for Cooperative Reinforcement Learning**
They claim a discrete communication is better. Agent communication through a broadcast-and-listen mechanism. They use PPO CTDE since is cooperative, with attention mechanism when summing the messages (i.d. Mine with attention module in merging messages)
GOOD: They use a self attention module: multiplicative attention of Lowe 2015
BAD: the algorithm used has limitation of applicability but can be used with mod in a different context
They claim that discrete performance of communication is lower than continuos (and probably interpretability in messages should be dropped when agent communicatinrg between themselves) also they propose as extension the use of Vaswani 2017 module of dot product attention.

### (4)
- **Learning Individually Inferred Communication for Multi-Agent Cooperation**
Learn with who you want to talk to reduce the overhead of communication in a collaborative setting. This is applicable to case where we have a critic centralized (join action-value), therefore it can be used also to reduce the communication in other appraoches like TarMac. They propose a two phase training in which the environment and network of action is trained first (or with whatever pre-trained CTDE algorithm) then the proposed network of prior for communication is learnt after.
BAD: each agent has to know who is around him in order to decide who speak with

### (4)
- **Learning Communication for Cooperation in Dynamic Agent-Number Environment**
CTDE but to limit the explosion of complexity in training use divide the agents in groups. It doeas use a dot product attention in order to decide which agent will become the leader of a dynamic group. Himself choose who to include in the group. Has a communication channel working through LSTM, selecting the important parts (?). Use QMix to assign accordingly the rewards portions to every contributor
BAD: only applicable to cooperation

### (4)
- **Multi-Agent Graph-Attention Communication and Teaming**
Further of the interesting new attempt to give decide who should speak with who by adopting a module which produces a graph of attention.

### (4)
- **Targeted multi-agent communication algorithm based on state control**
RQ: reduce the communication channel trasmission without loose in results
BAD: They believe that the communication should happen only under some events and therefore require a customization ad hoc of parameters.
"can broadcast communication messages only when the current communication message changes greatly in comparison to the previously broadcasted communication message or aftern n steps"
Double attention mechanism self and external to acquire only interesting messages internally. Buffer of past messages maintained in order to reason at every step with comm of agents even when they send nothing (nothing new). Tested on StarCraft

### (5)
- **Communication Efficiency in Multi-Agent Systems**
Talk about ACL message format for communication and how these can be integrated in a robotic context in a way to have better efficency. Use a second channel of communication called backchannel. It is interesting for the discussion about the necessity of a structure in the message which can be contrapposta all'uso di un messaggio in the continue space.

### (5)
- **Adversarial Attacks On Multi-Agent Communication**
Claim that we need to know the model of the attack in order to train our policy to prevent it. They do that for a gneral framework of attack by using having an agent acting the attacker which try yo get a surrogate perform wrost while training the surrogate to imitate the actor. Is just a training procedure framework in which we assume to know the other actor actions and improve itd policy.
RQ: 
GOD: the surrogate idea
BAD: homogeneous agents, works by broadcast and at all steps the agents exchange feature maps obtained from LIDAR views which are expensive.
We found that more practical transfer attacks are more challenging in this setting and require aligning the distributions of intermediate representations.

### (5)
- **Efficient Agent Communication in Multi-agent Systems**
address message passing and agent discovery in dynamic environments where agents arrive and go
BAD: Rely on an infrastructure of communication which in dynamic environments where there is not a center of communication and cannot exist a unique Actor Manager is not implementable

### (5)
- **IMPROVING COORDINATION IN SMALL-SCALE MULTI-AGENT DEEP REINFORCEMENT LEARNING THROUGH MEMORY-DRIVEN COMMUNICATION**
BAD: This as well as many other approaches memory driven rely on a shared buffer of communication where they read and write to infer a shared representation. Does not scale well and is not quite apllicable to a dynamic environment with agents which arrive and go. Also, they use centralized critic
Memory driven approach with MADDPG (that is actor critic CTDE)
We introduce a shared communication mechanism enabling agents to establish a communication protocol through a memory device M of pre-determined capacity M (Figure 1). The device is designed to store a message m ∈ RM which progressively captures the collective knowledge of the agents as they interact. An agent’s policy becomes µθi : Oi × M 7→ Ai , i.e. it is dependent on the agent’s private observation as well as the collective memory. Before taking an action, each agent accesses the memory device to initially retrieve and interpret the message left by others. After reading the message, the agent performs a writing operation that updates the memory content. During training, these operations are learned without any a priori constraint on the nature of the messages other than the device’s size, M. During execution, the agents use the communication protocol that they have learned to read and write the memory over an entire episode. We aim to build a model trainable end-to-end only through reward signals, and use neural networks as function approximators for policies, and learnable gated functions as mechanisms to facilitate an agent’s interactions with the memory. The chosen parametrisations of these operations are presented and discussed below.

### (5 aside)
- **The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning**
RQ: Can an agent with self interest be damaging for a team of coordinated agents in a common environment with shared communication channel?
They show empirically how agents which ignore others rewards can assume adversarial behaviours changing properly ther own internal state to make the others do what they want.
BAD: it is true only when the others are freezed, if they are learning all together this is prevented.

### (6)
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
-->
Interessante loro modello di attenzione e trajectory immagination but (Propongono come estensione) can be improved using it in better framework of message passing and info aggregation (cioè hai le intentions ora le devi usare per prendere buone decisioni)

### (6)
- **Learning to Ground Multi-Agent Communication with Autoencoders**
RQ: Should we first learn a common language of communication before expecting to communicqte with success
"The main challenge of learning to communicate in fully decentralized MARL settings is that there is no grounded information to which agents can associate their symbolic utterances. This lack of grounding creates a dissonance across agents and poses a difficult exploration problem. Ultimately, the gradient signals received by agents are therefore largely inconsistent."
<!--ehm boh, mi sembra molto strano...-->
BAD: they have a shared module for processing the communication messages which are all concatenated together. No attention. Also processed communication messages concatenated with image features encoded are passed through a GRU module to extract the policy action.
Also they use an autoencoder local for everyone to encode and decode the image from which extract the communication message.

### (6)
- **Learning Multiagent Communication with Backpropagation**
CommNet from 2016 is a first attempt of including a learned communication channel: there is a single controller which output actions for every agent and take in conmsideration all agent's states. It averages the hidden states for centralized communication
Is usable only in environmnet cooperative.
BAD: no structure just take all together and spit output htat can be doable in very easy MARL but when mthere are complex relations it become not scalable. Suffer of the credit assignment problem for multi agents.

### (6)
- **LEARNING WHEN TO COMMUNICATE AT SCALE IN MULTIAGENT COOPERATIVE AND COMPETITIVE TASKS**
IC3Net as extension of CommNet as well use a continuos vector
Averages the hidden states for centralized communication
Learn also to communicate only with some of the all individuals. Can be extended to mixsed but not scale well since the conflict of interests with multiple target
//BAD: They require a central aggregator for messages or a mechanism of sync to decide the action accordingly to all surrounding actors.//

### (6)
- **TarMAC: Targeted Multi-Agent Communication**
TarMAC introduce targeted communication to let coordination arise. The main part is a module of Multi head attention used to decide how much of the information coming from other to retain and consider. Both sender and receiver have to produce a similar signature in order to have higher consideration. This give a lot of responsabilities of choice to the receiver which unconscious of the info coming should decide to hear. (mine has ever the choice on the listener, more specifically on its coordinator, but this is trained separately to the policy used to move therefore can choose with more liberty)  
It state also about the necessity of continuos vector as said by CommNet and IC#Net, and centralized training since coordination is not enough in complex (like ad) scenarios.
Also we have to say that they used an approach of shared weights in their experiments therefore we have no guarantee that slightly different policies will fit in the behaviour of the group.

### (6)
- **Multiagent Bidirectionally-Coordinated Nets**
BiCNet / ATOC
Both use a bidirectional recurrent network as a communication channel. They fix the positions of agents in the bidirectional recurrent network to specify their roles.
CommNet is a single network designed for all agents. The input is the concatenatio of current states from all agents. The communication channels are embedded between network layers. Each agent sends its hidden state as communication message to the current layer channel. The averaged message from other agents then is sent to the next layer of a specific agent. However, single network with a communication channel at each layer is not easy to scale up.
More recent approaches for centralised learning require even more communication during execution: CommNet (Sukhbaatar et al., 2016) uses a centralised network architecture to exchange information between agents. BicNet (Peng et al., 2017) uses bidirectional RNNs to exchange information between agents in an actor-critic setting. This approach additionally requires estimating individual agent rewards.

### (6)
- **Learning Attentional Communication for Multi-Agent Cooperation**
ATOC attentional communication model to learn effective and efficient communication uunder decpomdp environment for large-scale MARL. An initiator agent has the ability to form a local group for the exchange of encoded observation for coordinated strategies. AC model with bidirectional LSTMs as communication channel. All with weights shared therefore this is suitable to large scale environments. Differently from BiCNet, use encoded observations and attentional communication model in a way to send the relevant information.

### (6)
- **Multi-Agent Concentrative Coordination with Decentralized Task Representation**
Coordination for reaching subtasks by enhancing the centralized q mixing network with environment information specific of the subtaks
BAD: not really partial observability

### (6)
- **Learning Efficient Multi-agent Communication: An Information Bottleneck Approach**
They treat the communication badwidth problem faced with a scheduler which should deliver a lot of messages probably redudant or unecessary therefore on the side of the scheduler an additional processing operation is done here to limit the unnecessary communications.
GOOD: to address the similarity of messages (and so be able to merge them) they make sure the messages respect an upperbound in the entropy

### (7)
- **Goal Consistency: An Effective Multi-Agent Cooperative Method for Multistage Tasks**
They have a set of predefined goals. A general observation value relative to environment and a vector of observations goals related. An attention module gives weights to select at each time which goal is the one that in relation with the current general observation is more important for the action selection.
BAD: being a CTDE the critic works with the whole vector of obs from all agents + all actions; there is no message passing during execution so this does not generalize, also they have all same goals and intentions so they should only coordinate on the order.
The use losses between probability distributions of different agents for consistency...
GOOD: having two soft attention modules (local and global/centralized) you have two probabilities distributions, you can use the second to influence the first

### (7)
- **Structured Diversification Emergence via Reinforced Organization Control and Hierarchical Consensus Learning**
RQ: how do the agents form teams spontaneously, and the team composition can dynamically change to adapt to the external environment? How to maintain diversity in the behavior of different teams and to form tight cooperation between the agents within the team?
Merges some learned parts with coded wanted constraints
Graph theory algorithm in first organizational module (not differentiable) to make adaptive teaming decisions. The hierarchical consensus module then generate the team intention and the individual intention based on the obtained teaming results. Finally, the decision module outputs the structured diversification policies
preformation of groups before reasoning over intention for policy?
they use the trick of global state during training to have better results...


### (6)
- **MULTI-AGENT COLLABORATION VIA REWARD ATTRIBUTION DECOMPOSITION**
CollaQ use my same idea of qalone + qcolla but use heavy attention with transformers before merging the observations
implicit reward assignment among agents
each agent has a decentralized policy that is (1) approximately optimal for the joint optimization, and (2) only depends on the local configuration of other agents
Qcollab part depends on nearby agents but vanishes if no other agents nearby
Differntlyfrom others they reach ad-hoc team play since other's learned Qi functions highly depend on the existence of other agents

### (1)
- **Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms**
overview

### ()
- **TOM2C: TARGET-ORIENTED MULTI-AGENT COMMUNICATION AND COOPERATION WITH THEORY OF MIND**
Being able to predict the mental states of others is a key factor to effective social interaction. It is also crucial for distributed multi-agent systems, where agents are required to communicate and cooperate. In this paper, we introduce such an important social-cognitive skill, i.e. Theory of Mind (ToM), to build socially intelligent agents who are able to communicate and cooperate effectively to accomplish challenging tasks. With ToM, each agent is capable of inferring the mental states and intentions of others according to its (local) observation. Based on the inferred states, the agents decide “when” and with “whom” to share their intentions.
SOTA on reward and communication efficiency
Target-oriented Multi-Agent Cooperation: communication about a target
Aassuming to know pose of every agent, infer the target goal of each one in relation to itself (self observation is taken and encoded with attention), in order to build a graph NN used to send messages. The messages received are used by internal planner to decide what to do.
1. Each agent first receives a local observation and encodes it with the encoder.
2. Then it performs Theory of Mind inference to estimate the observation of others and predict their goals.
3. Next, it decides ‘whom’ to communicate with according to local observation filtered by the inferred goals of others.
4. In the end, the planner in decision maker outputs the sub-goal according to what it observes, infers, and receives.

### ()
- **Multi-Agent Concentrative Coordination with Decentralized Task Representation**
MACC is a CTDE method which enhance coordination arising, do not use communication. Use a subtask internal representation to produce attention on his innternal trajectory and make better choices. On the centralized part there is a mixer which takes all states_i, trajectories_i and env_State

### (none)
- **Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition**
Our objective is to identify how to train a team to be resilient to different adversarial strategies.


## Ideas and notes
- An Idea could be group by intention the agents (clusters of latent space trained with proper loss)(the possibiliities of intention are limited so CTDE would be able to perform at eval time)
- "We can make a reasonable assumption: if two agents both have the same perception of the environment and the same goal, their behaviors should be closely coordinated."
- Bidirectional RNN allow to maintain the internal state of the agent e share info with collaborators however, it assumes that agents can know the global Markov states of the environment, which is not so realistic except for some game environments.
- Let's agree on who should go on first then it develop as hierarchical?
- What the messages should rewpresent: trajecories? own observations or request to the others?
- N agents take decisons? or one agent which control multiple actors?
- Infer the trajectories of other agents which avoid to communicate for better indirect coordination
- The communication has to be reduced to eliminate cluttering of channels: targeted comm or in broadcast choose when to communicate or choose when hear
- How to handle different goals simultaneously? both conflictual and not


**Boundaries to the research**

The environment is:
- partially observable, the single agent has not full knowledge
- dynamic, the environment evolve even without actions
- non stationary, evolve from actions of others which you cannot predict
- represent a non-zero sum game, the agents are not fully cooperative/competitive

The communication has to be:
- decentralized
- only A2A (not with infrastructure)

We shall decide if the communication has to be:
- Learnt via backpropagation or not? (the main aim is the comm or the effect of the comm)
- What to communicate?
- When to communicate?
- It should be targeted or broadcasted?
- Messages standardized (template-based, agent communication lang ACL)? Discrete or in the continuum?
- We should enforce the bandwidth limitation (scheduler, limits in messages exchanged)? 
- Focus on robustness (scheduler, delays, packet drops, link failures)?
- Focus on resilience from adversarial and self-interested agents?
- Focus on scalability and etherogeneous agents?
- Should consider approaches with centralized critic (training time) or go for fully decentralized?

## RQ
*These are the main ones but also the most "answered" with different approaches*

- How to do message communication inexpensive? (what communicate)
- How to reduce the trasmission of info useless? (when communicate)
- How to do optimal message exchange? (targeted communication or groups formation)

*These are more appealing to me*

- How to handle heterogeneous types of agents? (they should talk the same language even if they act with different probabilistic policies)
- How to defend themselves from a malicious agent or with self interests which communicate with us? <-- use unambiguous message as coordination which can avoid this case, maybe is too light as a problem
- How to do safe coordination and navigation even in presence of non collaborating agents? <-- apparently not faced
- How to achieve shared consensus on an optimal goal policy through only local computation and communication with neighboring agents? (distribuited, partial observability) <-- convergence to a shared goal via broadcast message passing and hierarchical RL in each agent to follow the goal

---


- Can agents learn autonomously when coordination is necessary and preferable to selfish behaviour through communication?  --> answer by quantitative results in different environments
- Is it possible to design a communication strategy that leads to an effective and efficient group coordination?
(define coordination communication effective and efficient e.g. n messages, rewards nell'env)


- Can the communication of messages between indipendent agents allow the formation of a shared consensus in mixed (cooperative/competitive) environments?       --> behaviour analysis
- Is the communication effective to reach coordination between indipendent agents?      --> ablation


---







> Still open questions? <!-- can be usefull for the conclusions? -->
Although powerful function approximators like neural networks can cope with continuous spaces and generalize well over large spaces, open questions remain like how to explore large and complex spaces suffciently well and how to solve large combinatorial optimization problems.



> (solo + extension): Can be effective anyway acting alone?  <!-- use in the ablation -->
Many recent papers focus on persistent and perfect communication between all the agents [9, 15, 35, 36]. This setting, while appealing, goes against the principle of autonomous agents, at least in its implementation, as the algorithms can be seen as centralized learning and execution with alternative neural network architectures. Furthermore, persistent and perfect communication is usually not available in the real world and it creates single points of failure if the communication goes down or an agent stops to communicate, as the rest of the agents would not be able to act.
[9] Abhishek Das, Théophile Gervet, Joshua Romoff, Dhruv Batra, Devi Parikh,
Michael Rabbat, and Joelle Pineau. 2019. TarMAC: Targeted multi-agent communication. In 36th International Conference on Machine Learning, ICML 2019,
Vol. 2019-June. 2776–2784. http://arxiv.org/abs/1810.11187
[15] Shariq Iqbal and Fei Sha. 2019. Actor-attention-critic for multi-agent reinforcement learning. In 36th International Conference on Machine Learning, ICML 2019,
Vol. 2019-June. 5261–5270. http://arxiv.org/abs/1810.02912
[35] Amanpreet Singh, Tushar Jain, and Sainbayar Sukhbaatar. 2019. Learning when
to communicate at scale in multiagent cooperative and competitive tasks. 7th
International Conference on Learning Representations, ICLR 2019 (12 2019). http:
//arxiv.org/abs/1812.09755
[36] Sainbayar Sukhbaatar, Arthur Szlam, and Rob Fergus. 2016. Learning multiagent communication with backpropagation. In Advances in Neural Information
Processing Systems. 2252–2260. https://arxiv.org/abs/1605.07736


> May an indipendent learner (different training) work better? <!-- possible ablation extention -->
Independent learners The naïve approach to handle multi-agent problems is to regard each agent individually such that other agents are perceived as part of the environment and, thus, are neglected during learning. Opposed to joint action learners, where agents experience the selected actions of others a-posteriori, independently learning agents face the main difficulty of coherently choosing actions such that the joint action becomes optimal concerning the mutual goal (Matignon et al. 2012b). During the learning of good policies, agents infuence each other’s search space, which can lead to action shadowing. The notion of coordination among several autonomously and independently acting agents enjoys a long record, and a bulk of research was conducted in settings with non-communicative agents (Fulda and Ventura 2007; Matignon et al. 2012b). Early works investigated the convergence of independent learners and showed that the convergence to solutions is feasible under certain conditions in deterministic games but fails in stochastic environments (Claus and Boutilier 1998; Lauer and Riedmiller 2000). Stochasticity, relative over-generalization, and other pathologies such as non-stationarity are the main causes of failures.


> What is the sense in communication? 
Despite the plethora of works that analyze emergent behaviors and semantics, many works propose methods that endow agents with communication skills. By expressing their intension, agents can align their coordination and find a consensus (Foerster et al. 2016). Agents that are able to communicate can compensate for their limited knowledge by propagating information and fill the lack of knowledge about other agents or the environment therefore eluding the non-stationarity. Moreover, agents can share their local information with others to alleviate partial observability (Foerster et al. 2018b; Omidshafei et al. 2017).


> Coordination: integrations
Successful coordination in multi-agent systems requires agents to agree on a consensus (Wei Ren et al. 2005). In particular, accomplishing a joint goal in cooperative settings demands a coherent action selection such that the joint action optimizes the mutual task performance. Cooperation among agents is complicated when stochasticity is present in system transitions and rewards or when agents observe only partial information of the environment’s state. Mis-coordination may arise in the form of action shadowing when exploratory behavior influences the other agents’ search space during learning and, as a result, sub-optimal solutions are found. Therefore, the agreement upon a mutual consensus necessitates the sharing and collection of information about other agents to derive optimal decisions. Finding such a consensus in the decision-making may happen explicitly through communication or implicitly by constructing models of other agents. The former requires skills to communicate with others so that agents can express their purpose and align their coordination. For the latter, agents need the ability to observe other agents’ behavior and reason about their strategies to build a model. If the prediction model is accurate, an agent can learn the other agents’ behavioral patterns and direct actions towards a consensus, leading to coordinated behavior. Besides explicit communication and constructing agent models, the CTDE scheme can be leveraged to build different levels of abstraction, which are applied to learn high-level coordination while independent skills are trained at low-level.


> Mixing Network: integration
In a multiagent system, all agents share a global reward function. Once an agent learns some useful strategies earlier, the rest will choose lazier strategies, making the overall reward decline. To solve credit assignment among agents caused by all agents sharing a reward function, we introduce a mixing network from QMIX. The mixing network input is the local Q function of each agent, and the output is the global Qtot. Since each agent only depends on local observations and may not accurately estimate its local Q function, QMIX needs to take the states as an additional input to the mixing network.


> Distributed applications: integrations
[[An Overview of Recent Progress in the Study of Distributed Multi-Agent Coordination]]
In distributed control of a group of autonomous vehicles, the main objective typically is to have the whole group of vehicles working in a cooperative fashion throughout a distributed protocol. Here, cooperative refers to a close relationship among all vehicles in the group where information sharing plays a central role. The distributed approach has many advantages in achieving cooperative group performances, especially with low operational costs, less system requirements, high robustness, strong adaptivity, and flexible scalability, therefore has been widely recognized and appreciated. The study of distributed control of multiple vehicles was perhaps first motivated by the work in distributed computing [1], management science [2], and statistical physics [3]. In the control systems society, some pioneering works are generally referred to [4], [5], where an asynchronous agreement problem was studied for distributed decision-making problems. Thereafter, some consensus algorithms were studied under various information-flow constraints [6]–[10].


> Real communication: integrations
Traditional cellular communication follows an uplinkdownlink topology, regardless of the end-device’s location. However, LTE release 12 and 5G also supports device-to-device (D2D) communications, where physically close devices, e.g., two vehicles, can communicate directly over a socalled sidelink. Compared to regular uplink-downlink communication, D2D communications benefits from a shorter link distance and fewer hops, which is beneficial from a reliability perspective. Moreover, since communication is direct, i.e., without intermediate nodes, D2D has the potential to provide very low latency.


> Transition to a more complex environmnet: CARLA
The field of MARL has often dealt with rather simple applications, usually in toy-world scenarios or drawn from game theory and mostly with only a few (typically two) learning agents involved. We think this simplification makes sense and is helpful for getting a better understanding of principle possibilities, limitations, and challenges of MARL in general and of specific MARL techniques in particular. But it is not sufficient. In addition, the MARL field needs to focus more than it currently does on complex and more realistic applications (and is mature enough for doing so) for two main reasons. First, eventually this is the only way to find out whether and to what extent MARL can fulfill the expectations in its benefits; and second, this is the best way to stimulate new ideas and MARL research directions that otherwise would not be explored. There is a broad range of potential real-life application domains for MARL such as ground and air traffic control, distributed surveillance, electronic markets, robotic rescue and robotic soccer, electric power networks, and so on. Some work has already successfully explored the integration of learning in these real- world domains and has shown promising results that justify a stronger focus by the community on complex systems;


> Why RL suites AD applications?
Situational uncertainties are even more challenging. These will occur when autonomous vehicles begin to interact with other vehicles and external moving objects — for example pedestrians, bicycles, and animals. While being a highly complex perception and control system, the embedded system has limited knowledge about drivers’ behaviors, unusual events, erratic events, and most importantly lacks the semantic understanding of situations that humans deal with in their everyday driving. Multi-agent reinforcement learning is the modern framework to approach these types of problems (K. Tuyls 2012). A multi-agent system is a group of interacting agents (autonomous entities) sharing a common environment, sensing and acting on it at the same time. Due to the complex and frequently changing environment in autonomous driving, the agents cannot be fully programmed in advance. They will need to learn by trial and error. The challenge is that the errors should have minimal or no consequences — difficult to accomplish with a large moving heavy vehicle. Examples of these situations are merging in roundabouts or encountering objects with uncertain context, like a ball suddenly appearing — will there be a child running for the ball in front of the car? Internet of things (IoT), vehicle to vehicle (V2V), and vehicle to infrastructure (V2I) communications will play a key role in managing situational uncertainties but nonetheless the algorithms have to be designed with enough generalization and strong coordination abilities to handle those events.

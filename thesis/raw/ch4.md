## Method

- what you are gonna present
- what is the main movement of reasoning and critique about that
- my claim idea motivation
- how I intend to treat the problem
- brevissima spiegazione degli highlights della pipeline
- Presentation of the mathematical claim backing up the approach
- spiegazione dei moduli come pezzi a se stanti
- (nell'immagine della pipeline spieghi il giro)
- 


<!--what you are gonna present-->
This chapter will introduce the method proposed to solve the described problem of coordination. Grounding its theoretical basis on sound algorithms and common knowledge, this project work face the problem with a different approach apporting important contributions in terms of algorithmic implementation of the reasoning process for MARL agents. We back our claim with extensive test in different environments and ablation study measuring the impact of each component in the architecture, finally the algorithm is tested in a complex environment to show the results obtained in a highly realistic simulated environment.


we resort to an end-to-end learning of Qi for each agent i with proper decomposition structure inspired by the theory.
To see this, we expand the Q-function for agent i: Qi = Qi(si, ai,ˆri) with respect to its perceived reward


### reasoning approach to coordination
A common approach to this problem is to see the coordination as the main necessity in evaluating what action to intraprend in the environment. This can intuitively result as a profitable choice when the main objective is the cooperation but when agents are allowed to assume different behaviour to reach self evaluated target and objective this method of reasoning can become harmful.

An intuitive example of this situation lended from the real world is a car driver willing to go from a starting point to its destination. If at each "step" he should consider the intentions and actions of other drivers, evaluating the possibility of let someone else go forward of take the lead the trip would become endless. This reasoning is accentuated in large environments where reward signals can be sparse, becoming harder to extract rules of the environment and the exponential possiiblity of interactions with others can lead to high uncertainty of action and a detriment in learning when prioritizing exploration or suboptimal convergence.

The second critique is about the necessity of complex dialogues to enstablish coordination. The objective of a communication channel should be the one of reducing uncertainty and non stationarity effects providing additional information about others, but many methods instead rely on exchanges of complex vectors encapsulating their history or personal thought used to act. With a complete disentaglement between the action selection and the communicated information an agent can be unable to fully understand the true will of the speaker in a general context. Also complex environmnets are much more intractable in terms of state evaluation therefore is important to keep the size of state space small providing the agents of only the information useful for their reasoning process.

As final tought the work proposed do not claim necessary a declarative and imperative agreement between agents since coordination can arise from continued interactions in the environment. Indeed, being subject to the same environment rules, all agents in the end will avoid decisions that will damage themselves and will collaborate occasionally in order to maximize the respective reward signals.

On top of this what we propose is a Q network which can act both following an egoistic will that on time can act with preference for a more altruistic behaviour; a reasoning process able to include the others information, wisely evaluating what to do on the base of new knowledge obtained from a specific agent; a simple still effective communication channel for exchanging of information. As finall touch we implement both coordination and policy modules with recurrent nn to maintain consistency in the choices over time and train the whole in a centralized training decentralized execution (CTDE) paradigm. The architecture in the whole is represented in figure X. Every module is discussed in detail in the subsequent paragraphs.



    This is the part of the architecture which is basesd on the decison of the single agent and will expand such decision incorporeating collaborations aims decided in a secondary module not properly linkaed to the primary decision of action. Finally an important touch in the creation of the arcitecture, all modules involved in taking decisions where implemented with recurrent modules in a way to obtain time consistency between decisions.

    My approach instead, takes always in consideration the selfish will and then evaluate the possibilty of modifying the behaviour to cooperate or compete with others.

### action policy
The Q policy network is in charge of predicting the state-action value for each action of an agent on the base of the info at his disposal. We can define it with the following formula <> where oi, ai and hi are respectfully the observation, action and hidden state of agent i. This could be considered enough for an implementation of an indipendent learner, nevertheless we consider in this setting communication messages as additional information that when available could aid the action decision process. With this intent we reformulate the previous definition as the sum of two terms <>. Here Qself resemble the selfish action intention. When computed it produces state-action values for the current state and update the hidden state for the next state. The second term Qcoord takes the current updated hidden state and incoming messages filtered out by a coordination module. Furthermore we introduce the use of a feature extractor that executed on the raw observation allow to extract meaningfull information and let the policy network reason in a more constrained space. To note that the input processing module used has shared weights between all agents. This decisions as highlighted by [] is an important implementation detail to allow all agents reason on the same distribution of data.
The final formula representing the policy network involved is the following:
<>
The two terms are described as following and implemented with two GRU sharing only the hidden state. The recurrence of the first is necessary to compute the new observation in relation with the past while the second to properly mix the self interests with the new one obtained from third agents:
<>

### coordination module
The coordination module aim at determine the relevance of the incoming messages with respect the self intentions in order to output a mask of coordination used to filter out incoming messages <>. In this context we refer to a message as the communicated intention of an agent to act in a certain way to fulfill a personal objective. It is described by the tuple <>.
As already proposed by [ATOC] the module in charge of such reasoning is implemented through a BiGRU layer considering all together a composition of self and others intentions. Each individual score produced then is used to obtain an indipendent probability of communication with a two way softmax.

### mixer network of control
We encapsulate such approach of coordination in a learning framework which respect the CTDE paradigm. All qs and observations of the agents are delivered to a centralized state-action network restricted to the computation of a monotonic value function which factorize the singles state-action values.
<>
This mixer network implementation was introduced by [Rashid] and its a well known framework that despite the limitation in representation ability represented a SoTA approach in the field.

    we shall say that our implementation is not constrained in QMIX framework and can be implemented as well in other learning frameworks.


### Training supervision and losses components
The Q policy network is trained end to end with the gradient errors being able to flow between the two modules by mean of the hidden state shared. The rule of update follow the common implementation of the QMIX [Rashid] where a temporal difference rule of update is being involved.
<>
The coordination part instead involve more creativity in designing an efficient learning schema. Since the probability distribution produced by the single agent for each other see his effects in a modified set of values for the state-action pairs, we can effectively measure such improvement or decrease in the performance assuming an optimal estimate of Q value from the policy. We therefore define the formula of update of the gradient being a clipped delta between the Qs obtained with the predicted mask of coordination with respect to the estimated values obtained from inverted probabilities.
<>
here in the formula we intend.... The adoption of the QMIX approach give us an additional element which can be used to better estimate the gain in using a mask of coordination in place of another. Being the mixer network a mapping from states to a set of weights in the hidden space used in linear combination with q estimated indipendently per each agent, we can reuse this parameters to weight the qs used to criticize the choice of adopting a certain communication mask. In this terms we have updates more reliable and reletable to the exact gain given by the communication with a certain agent in a certain state. We find mention of this in the above formula with the <> term, computed as follow <>

### Training details and hyperparameters
The training of pipeline has been executed introducing randomicity and exploration in action selection by adopting an epsilon decay technique scheduled to go from 0.9 to 0.05 in 60% of the training time. Furthermore to successfully exploring also the communication possibilities among the probability distribution predicted it has been adopted a Gumbel Softmax function which exponentially separates the output logits values introducing randomness from a normal distribution in the process. The optimization process needed particular attention in order to not overfit the situations at hand and end up in catastrophic forgetting situations. To avoid that, a regularization parameter of weight decaying was used in the optimizer. The learning process include also the use of target networks which was tuned and tested opting for a soft update every step.
Size of the networks, learning rate, and decay parameters were seelcted carefully for all environments to obtain the best out of the algorithm.
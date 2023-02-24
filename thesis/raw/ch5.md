## Evaluation
The aim of this chapter is to evaluate the proposed method and assess its efficacy in addressing the research questions outlined previously. The evaluation process will comprise of a comparative analysis with relevant and pertinent methods in the field, followed by the presentation of results obtained through a systematic and comprehensive evaluation methodology. The results will be analyzed in detail, including an ablation study of the method's components and implementation details, to identify its strengths and limitations. The ultimate objective of this evaluation is to furnish the reader with a thorough understanding of the method's performance and facilitate the researcher in drawing meaningful conclusions about the effectiveness of the proposed solution.

### Baseline of comparison
- Individualized Controlled Continuous Communication Model (IC3Net)[] propose it's approach as extension of a previous method [], introducing a gating mechanism on each other agent's channel of communication. They use the current observation encoded both as internal thought and communication message, then, following a mask of communication obtained in the previous timestep, the same vector is processed through an LSTMCell module with all incoming messages averaged together. The result is used as input of two output heads to obtain the action values and the individualized probabiity of communication.
<!--a message-generation network is defined at each agent and connected to other agents’ policies or critic networks through communication channels. Then, the message-generation network is trained by using the gradient of other agents’ policy or critic losses.-->
- ATOC[] is an Attentional Communication model proposed to learn effective and efficient communication at scale by adopting weight sharing between every agent's network. In their proposal the communication message is represented by the hidden state of a recurrent module processing the observation at each step, therefore sharing a vector resembling an history of the agent. On the base of the internal thought the agent will decide if communicate with his neighbours or not, noticing that this decision will create a group of maximum $m$ agents maintained for $T$ steps, with $m$ and $T$ as hyperparameters. The messages are then mixed with a BiLSTM module and merged with the internal thoughts to obtain the action values.
<!--ATOC focus on how to improve the coordination by not using simple observations as message but hidden statesand using a bidirectional layer mix those for those agents partecipating in a group of communication. The groups of communication are enforced for T step to maintain recurrence over the decisions and for stability of communication.-->

### Environments and metrics
For the environment we will consider the envioronment of Pettingzoo pursuit-evaders which is comparable to the predator prey problem with an enforced necessity of cooperation. 4 agents at a time need to cooperate to capture an evader. The environmnet can be briefly described as a grid world where some agents follow other entities to capture them. Rewards are delivered for "tagging" an evader that means be in the same position of it, or "capturing" it which require at least 4 pursuer to surround the evader. 
The second environment we will consider is Switch4 a grid world were 4 agents given only their position coordinate they have to find in the envirnmnet the respective switch which also represent the only position were they receive a reward. The difficulty is introduced by the conformation of the environment. Two rooms where agents spawn are connected by a small corridor hall which can be crossed by only an agent at a time therefore coordination is necessary otherwise agents will remian stuck trying to cross all at the same timein order to reach the wanted position. To note also that in this case normally the agents cannot see the others agents therefore they can only infer from the evolution of their state given the action choosed.

### Results


- 1. (first step) encoding + self thought GRU to produce plan
- 2. (come li filtro in ascolto/decido se parlo) [mio plan+loro plan] in BiGru produce attention
- 3. (come li mixo con la mia volontà) starting from self hidden mix sequentially the other (masked) plans with GRU
- 4. (mix --> azione) [self Q + Coord Q] --> azione


- Individualized Controlled Continuous Communication Model (IC3Net), in which each agent is trained with its individualized reward and can be applied to any scenario whether cooperative or not.
- IC3Net uses a gating mechanism that allows agents to block their communication with others.
- IC3Net as extension of CommNet as well use a continuos vector (agent hidden) but it has gating mechanism to decide if or not communicate and use an LSTM head for each agent instead of all.
- Weight sharing for all net apart from the LSTM nets: (i) allow the model to be applied to both cooperative and competitive scenarios (ii) better distribution of the rewards, solving credit assignment problem
- Averages the hidden states for centralized communication
- H+c to the lstm to compute the action
- Can be extended to mixed but not scale well since the conflict of interests with multiple target
- 1. self linear feature extractor process obs and is called hidden_state
- 2. matrix di hidden filtered by dones and by prev decision of communication, average per ogni riga = lorothought (**mix tra loro poco informativo**)
- 3. [mioencoding+lorothought] LSTMCell starting with hidden as hidden state and cell state recurrent = mix (**stessi dati sia azione che decisione di comunicare**)
- 3. mix --> due head (azione, comunica si no per tutti gli agenti) (**hardcoded in the net how many at max**)


- ATOC, attentional communication model to learn effective and efficient communication under decpomdp environment for large-scale MARL.
- An initiator agent form a local group for the exchange of encoded observation for coordinated strategies.
- AC model with bidirectional LSTMs as communication channel.
- All with weights shared therefore this is suitable to large scale environments. Differently from BiCNet, use encoded observations and attentional communication model in a way to send the relevant information.
- Exploit a bidirectional LSTM unit as the communication channel to connect each agent within a communication group. The LSTM unit takes as input internal states (i.e., encoding of local observation and action intention) and returns thoughts that guide agents for coordinated strategies. Does similar as me but on the side of the sender not of the receiver
- 1. self linear feature extractor process obs and **sharing of LSTM hidden i.e. history**
- 2. hidden are processed by attentional module to determine (**on the base of self thought**) if I **should talk with all possible neighbours** (**capped at m**), then **kept for T** steps
- 3. masked hidden are mixed BiSequentially in LSTM starting from your thoughts = lorothought
- 4. [mieithought.lorothought] --> azione


# TODO ATOC https://opendilab.github.io/DI-engine/12_policies/atoc.html#quick-facts
# TODO Different from ATOC, TarMAC (Das et al. 2019) uses attention in communication to decide who to communicate with. TarMAC is interpretable through predicted attention probabilities that allow for inspection of which agent is communicating what message and to whom. Additionally, TarMAC proposes multi-round communication where agents coordinate via multiple rounds of communication before taking actions in the environment.


    With this implementation we claim the ineffectiveness of forcing a collaborative behaviour at each step as in [IC3Net]

### Metrics



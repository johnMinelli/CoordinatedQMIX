## Ablation raw notes
- SWTICH
mentre inverse e no_w arrivano alle stesse performance di coordinazione (goods) ma più tardi. questo hinder the learning process che quindi si attesta a risultati (tot_reward) minori
no_w hanno spike di loss molto alti durante il training che invece si presenta molto più stabile nella optout baseline ma anche in true/inverse (PURSUIT TOO)

note how true in this environment obtain better rewards (but still is down in success in the graph since the following). While initially the baseline oputout obtain some success from the agents, true seems to obtain a success from everyone after a period (4k) of learning and from that point onward the rewards are high (very fast in arriving at destination).

(in PURSUIT isntead does fit the objective)
true and Coord w/o weights are the ones performing better after the baseline while no Q_coord mostra evidenti effetti di mancanza di coordinazione nei risultati raccolti  

true sembra ci siano problemi a stabilire chi deve passare nel corridoio
inverse the problem is in two agents policy which are not good in going to target but coordination is fine
solo_qs venono fatte molte più azioni a vuoto (più incertezza) ed anche quando gli agenti sono uno di fronte all'altro non si coordinano bene e.g. uno non si sposta per far arrivare l'altro nel rispettivo switch
no_w perchè vanno tutti nello switch di qualcun'altro?


- PURSUIT
baseline works much better
also True and inverse not bad this since inverse is just a simplified learning and True does fit the environment since having info from everyone is not a bad idea here since everyone has the in general (not specifically since maybe want to catch one different to the others) the same objective
again no_coordination do not work good


optout use strategies like moving along the edges.
solo_qs
they spread out and do not stay put together. so the catch happen only in some rare case they spawn near
no_w
are very compact in corner, rarely explore but if they move they move all together
true
are not able to coordinate properly and rarely do a capture even if a pursuer get near the grop. The group anyway is very cohese
inverse
they have zero interest in exploration of the space


## Ablation structured notes
'switch' env
- Results ('tot_reward' statistics):
'baseline' : 1°
'inverse' : 4°
'true' : 3°
'no_w' : 2°
'no_coord' : 5°

- Analysis results:
  - 'inverse' and 'no_w' are able to demonstrate same coordination performances (indicated by 'goods' statistic) as the 'baseline', but since they necessitate more time in learning the results show a lower 'tot_reward'   
  - observing the training loss of the Coordinator module we see very high spikes in 'no_w' while adopting 'true and 'inverse' we observe a learning much more stable (maybe stable because ineffective)
  - 'true' obtain very good results in the end but the process, seems not to be linear. While the strategy of the baseline allow agents to incrementally learn and find their way in the environment, 'true' do not report successes initially, then agents after learning how to succeed all together, start obtaining rewards and optimize the stragey. This is probably because we are incentivizing the learning to consider everyone intention instead of considering the single situations of understanding in the environment. (PURSUIT everyone is needed in this small env)
  - 'no_coord' show an evident lack of coordination since the results obtained

- Analysis strategies:
  - 'true': seems there are problems in enstablishing who should pass first in the corridor and this restrain some agents to arrive in time and get a good 'tot_reward'
  - 'inverse': coordination seems to work fine but two out of the four agents are not well trained and show problem in reaching the respective target
  - 'no_coord': many actions forward and backward without an effective result are performed by agents. Seems they are disturbed by other agents' actions and can not understand/anticipate each other movements. Even when they are placed one in fron the other they are not able to coordinate.
  - 'no_w:' everyone aim wrongly the target of another agent instead of their own


'pursuit' env
- Results ('capture' statistics):
'baseline' : 1°
'inverse' : 2°
'true' : 3°
'no_w' : 4°
'no_coord' : 5°

- Analysis results:
  - the 'baseline' works evidently better then others evaluated strategies
  - 'inverse' and 'true' in particular are not so far from the baseline. The first being a lighter (in terms of computational cost) version of the 'baseline' while 'true' since in this case, everyone information is useful for performing a 'capture' and get better results
  - observing the training loss of the Coordinator module we see very high spikes in 'no_w' while adopting 'true and 'inverse' we observe a learning much more stable (maybe stable because ineffective)
  - 'no_coord' show an evident lack of coordination since the results obtained

- Analysis strategies:
  - 'baseline' adopt strategies in acting like moving along the edges.
  - 'no_coord': the agents spread out and do not stay put together. The event of 'capture' happens only in some rare case like when they spawn near each other
  - 'no_w': stay very compact in a corner. The evident problem here is a rare demonstration of will of explore the space to perform 'capture', but if they move they move all together
  - 'true': they are not able to coordinate properly and rarely do a 'capture' even if presented with the opportunity (a prey get near the group of predators). The group anyway is very cohesive
  - 'inverse': the evident problem here is a missing will of explore the space to perform 'capture'



## comm/analysis and Interpretability
To conclude the evaluation analysis we observe the communication mechanism of CoMix by analyzing the evolution of the communication masks predicted and extracting useful insights. Comparing the state action values used during the computation of the loss of the coordinator we can indeed determine if the choice of coordination resulted as successful with respect the alternative coordination mask. Figure X show such good/bad ratio of coordination. That means if the loss measure the magnitude of our error the ratio express how many times we are correct in the prediction. Values increasing for all ablated strategies show promising results, with the base method obtaining the maximum value. In any case we should note that the learning process of coordination should preferribly be choosen respect the environment dynamics to obtain the best results e.g. in a fully cooperative environment could be more proficient train against the maximum amount of information at disposal and then learn skimming from these what is not useful at the current step. Another useful piece of information which can be obtained is the number times agents decide to coordinate with others. Over the time of training matching the intuition we observe a decrease in positive time of coordination in the switch environment while this happen in a much less pronunciated manner in Predator-Prey.
<!--TODO should i putt the graph? Should I divide by ablation flavours? the description is valid for optout and slightly also for no_w--> 
As a final note, CoMix implementaion provide intrinsic interpretabilty in the choice of action by the agents since we can reconduct an action to single interactions with other agents or due to reach self imposed objectives. For instance in the Switch environment when a single agent is left, its actions are not affected by others and $Q_coord$ become 0. In the case of Predator-Prey environment instead we can observe the norm of $Q_self$ and $Q_coord$ to determine if the agent is acting primarily following its will or adopted a strategy towards coordination.



To summarize, our analysis of CoMix's communication mechanism revealed valuable insights. We determined the success of coordination by comparing the state action values used during the computation of the loss of the coordinator with respect to the alternative coordination mask. The good/bad ratio of coordination, as shown in Figure X, indicates how many times our predictions were correct. The values increasing for all ablated strategies showed promising results, with the base method obtaining the maximum value. However, we noted that the learning process of coordination should be better tailored to the environment dynamics to obtain the best results.

Another useful finding was the number of times agents decided to coordinate with others. Over the time of training, we observed a decrease in positive time of coordination in the Switch environment, while this decrease was less pronounced in Predator-Prey. This finding matches the intuition that coordination is more challenging in the Switch environment due to the constantly changing tasks.

Furthermore, the implementation of CoMix provides intrinsic interpretability in the choice of actions by the agents. We can trace an action back to single interactions with other agents or due to the agents' self-imposed objectives. For instance, in the Switch environment, when a single agent is left, its actions are not affected by others, and $Q_{coord}$ becomes 0. In the case of Predator-Prey environment, we can observe the norm of $Q_{self}$ and $Q_{coord}$ to determine if the agent is acting primarily following its will or adopted a strategy towards coordination. Overall, our evaluation analysis of CoMix's communication mechanism provides useful insights into the coordination behavior of multi-agent systems.




### Limits
Training times


Despite the promising results presented in this paper, we acknowledge some limitations of the CoMix approach. One of the main limitations is the slow training process compared to the other methods used as a comparison. Although we achieved good performance, we had to invest significantly more time and computational resources in the training phase. We believe this is due to the inherent complexity of the approach and the number of trainable parameters. Another limitation is the specific implementation of the approach, which does not include mechanisms to limit the computation, such as restricting the range of attention of each agent. However, we note that these limitations can be overcome by implementing additional mechanisms on top of the CoMix approach, taking inspiration from the vast literature on multi-agent coordination. Overall, we believe that the CoMix approach represents a promising direction for multi-agent coordination, and we hope that our work will inspire further research in this field.


### Conclusion





In conclusion, we have proposed CoMix, a new coordination mechanism for multi-agent systems based on a combination of self-interest and coordination. We have shown that CoMix is able to achieve good results in two different environments, the Switch and the Predator-Prey. Our evaluation analysis has provided insights into the mechanisms underlying CoMix and demonstrated its effectiveness in terms of coordination and individual performance. Specifically, our results show that CoMix outperforms alternative coordination mechanisms and that it is able to learn to coordinate efficiently even in the presence of complex dynamics and partial observability. Additionally, our ablation study has provided useful information about the role of different components of CoMix and highlighted the importance of careful design choices in the learning process.

Overall, we believe that CoMix is a promising approach to coordination in multi-agent systems, and that it has the potential to be applied in a wide range of settings, from robotics to social networks. Future work will focus on further improving CoMix, for example by exploring different architectures and training strategies, as well as applying it to more complex scenarios. We also plan to investigate the interpretability of CoMix and its ability to learn human-like coordination strategies. We believe that our work represents an important step towards developing more intelligent and flexible multi-agent systems, and we look forward to further research in this area.
## Ablation notes
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



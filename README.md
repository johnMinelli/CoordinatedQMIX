# Learning a communication which incentive coordination

#### CARLA Traffic control with comunicating agents
#### Traffic control for multi agent autonomous driving

## Usage
To train 

```
python train.py
```

To evaluate:
```
python eval.py  --load_path ./eval/
```


train (switch)	`--name switch_no_coord --render_mode human --env CoMix_switch --gpu_ids -1 --coord_mask_type optout --batch_size 511 -ep 2000 --hi 128 --hc 64 --hm 32 --hs 1 --lambda_q 10 --seed -1`

train (pp4)	`--name pp_4 --render_mode none --env CoMix_predator_prey_4 --gpu_ids -1 -vi 5 -ve 1 --coord_mask_type optout --batch_size 512 -ep 5000 --coord_K_epochs 0.02 --K_epochs 0.02 --hi 128 --hc 128 --hm 32 --hs 1 --lr_co 0.00005 --lr_q 0.0001 --wandb --update_target_interval 40000 --chunk_size 10 --seed -1`

train (pp8)	`--name pp8_no_comm --render_mode none --env CoMix_predator_prey_8 --gpu_ids 0 -vi 5 -ve 1 --coord_mask_type optout --batch_size 32 -ep 5000 --coord_K_epochs 0.02 --K_epochs 0.02 --hi 128 --hc 64 --hm 32 --hs 1 --lr_co 0.00005 --lr_q 0.0001 --update_target_interval 40000 --wandb --chunk_size 20  --seed -1`

train (pp16)	`--name pp16_no_comm --render_mode none --env CoMix_predator_prey_16 --gpu_ids -1 -vi 5 -ve 1 --coord_mask_type optout --batch_size 32 -ep 5000 --coord_K_epochs 0.02 --K_epochs 0.02 --hi 128 --hc 64 --hm 32 --hs 1 --lr_co 0.00005 --lr_q 0.0001 --update_target_interval 40000 --wandb --chunk_size 20  --seed -1`

train (transport1)	`--name transport1_no_comm --render_mode none --env CoMix_transport_1 --gpu_ids -1 -vi 5 -ve 1 --coord_mask_type optout --batch_size 512 -ep 5000 --hi 128 --hc 64 --hm 32 --hs 1 --lambda_q 10 --update_target_interval 20000 --lr_co 0.00005 --lr_q 0.0001 --wandb --seed -1`

train (transport2)	`--name transport2_no_comm --render_mode none --env CoMix_transport_2 --gpu_ids -1 -vi 5 -ve 1 --coord_mask_type optout --batch_size 512 -ep 5000 --hi 128 --hc 64 --hm 32 --hs 1 --lambda_q 10 --update_target_interval 30000 --lr_co 0.00005 --lr_q 0.0001 --wandb --seed -1`

train (transport4)	`--name transport4_no_comm --render_mode none --env CoMix_transport_4 --gpu_ids -1 -vi 5 -ve 1 --coord_mask_type optout --batch_size 512 -ep 600 --hi 128 --hc 64 --hm 32 --hs 1 --lambda_q 20 --update_target_interval 40000 --wandb --seed -1`

fine tune (switch)	`--name switch_fn --render_mode none --env CoMix_switch --gpu_ids -1 -vi 5 -ve 1 --batch_size 16 -ep 3990 --K_epochs 0.02 --coord_K_epochs 1 --hi 128 --hc 128 --hm 32 --hs 1 --lr_q 0.000001 --lr_co 0.00005 --chunk_size 1 -wd 0 --min_buffer_len 1000 --max_buffer_len 10000 --continue_train 500 --fine_tune 1 --wandb`

fine tune (transport)	`--name transport_2_fn --render_mode none --env CoMix_transport_2 --gpu_ids -1 -vi 5 -ve 1 --batch_size 16 -ep 3990 --lr_q 0.000001 --lr_co 0.00005 --continue_train 1600 --fine_tune 1 --wandb`

fine tune (pp)	`--name pp_4_fn2 --render_mode none --env CoMix_predator_prey_4 --gpu_ids -1 -vi 5 -ve 1 --batch_size 16 -ep 3990 --K_epochs 0.02 --coord_K_epochs 0.02 --hi 128 --hc 128 --hm 32 --hs 1 --lr_q 0.000001 --lr_co 0.00005 --chunk_size 10 --continue_train 1450 --fine_tune 1 --wandb`

eval (switch)	`--gpu_ids -1 --models_path save --render_mode none --env CoMix_switch --name switch_fn --model_epoch 580 -ve 100`

eval (pp)	`--gpu_ids -1 --models_path save --render_mode none --env CoMix_predator_prey_4 --name pp_4_fn2 --model_epoch 2050 -ve 100`

eval (transport)	`--gpu_ids -1 --models_path save --render_mode none --env CoMix_transport_2 --name transport_2_fn --model_epoch 1950 -ve 100`
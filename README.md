# CFGDM: Energy-Free Guidance of Geometric Diffusion Models for 3D Molecule Inverse Design

Official code release for the paper Energy-Free Guidance of Geometric Diffusion Models for 3D Molecule Inverse Design

## Training the Models
We have a wrapper, which we use for batch launches as well as auto-scaling for your GPU availability. You can specify your hyperparameters in the `experiments/launch.py` file. The  `--resume`` flat is used to switch between training from a previous checkpoint or from scratch. 

If you want to independently launch training scripts, use: 

```
python main_qm9.py --exp_name single_cfg_PROPERTY_PLACEHOLDER \
                   --model MODEL_PLACEHOLDER \
                   --lr 2e-4 \
                   --nf 192 \
                   --n_layers 9 \
                   --save_model True \
                   --diffusion_steps 1000 \
                   --sin_embedding False \
                   --n_epochs EPOCHS_PLACEHOLDER \
                   --n_stability_samples STABILITY_SAMPLES_PLACEHOLDER \
                   --diffusion_noise_schedule polynomial_2 \
                   --diffusion_noise_precision 1e-5 \
                   --dequantization deterministic \
                   --include_charges False \
                   --diffusion_loss_type l2 \
                   --batch_size BATCH_SIZE_PLACEHOLDER \
                   --conditioning PROPERTY_PLACEHOLDER \
                   --dataset qm9_second_half \
                   --classifier_free_guidance \
                   --resume pretrained/cEDM_PROPERTY_PLACEHOLDER \
                   --guidance_weight GUIDANCE_WEIGHT_PLACEHOLDER \
                   --test_epochs EPOCH_REPLACE_PLACEHOLDER \
                   --class_drop_prob DROP_PROB_PLACEHOLDER \
                   --normalize_factors [1,8,1] \
```

**For multi-property launches,** use `launch_double.py` - the workflows should stay the same

## Evaluation
We also have a script for effective batch evals, which allows us to select between different checkpoints and guidance weights, `launch_evals.py,` which we use for distributing our evaling workflows. The flag --resume is used to differentiate between models trained from scratch versus those that are resumed. 

For individually evaluating a model, you case use the following placeholder: 

```
python eval_conditional_qm9.py --generators_path outputs/single_cfg_PROPERTY_resume \
                               --classifiers_path pretrained/evaluate_PROPERTY \
                               --property PROPERTY  \
                               --iterations 10  \
                               --batch_size BATCH_SIZE \
                               --task edm \
                               --use_wandb \
                               --exp_name EXP_NAME \
                               --override_guidance GUIDANCE_WEIGHT
```

## Qualitative evals
Finally, for qualitative evals, such as generating the figures used in the paper, you can use `launch_evals_qualitative.py`

For an individual run, use
```
python eval_conditional_qm9.py --generators_path outputs/single_cfg_PROPERTY \
                               --property PROPERTY  \
                               --iterations 10  \
                               --batch_size BATCH_SIZE \
                               --task qualitative \
                               --use_wandb \
                               --exp_name EXP_NAME \
                               --override_guidance GUIDANCE_WEIGHT \
                               --ckpt CKPT
```

## Credits
Our work is built on top of seminal work by two papers: 
* EDM: https://github.com/ehoogeboom/e3_diffusion_for_molecules
* EEGSDE: https://github.com/gracezhao1997/EEGSDE

## Citations

If you use this work (or code), then please cite either of these papers:

```
@inproceedings{
nagaraj2024energyfree,
title={Energy-Free Guidance of Geometric Diffusion Models for 3D Molecule Inverse Design},
author={Sanjay Nagaraj and Jiaqi Han and Aksh Garg and Minkai Xu},
booktitle={ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications},
year={2024},
url={https://openreview.net/forum?id=BsstqCIeOS}
}

@inproceedings{
garg2024energyfree,
title={Energy-Free Guidance of Geometric Diffusion Models for 3D Molecule Inverse Design},
author={Aksh Garg and Jiaqi Han and Sanjay Nagaraj and Minkai Xu},
booktitle={ICML 2024 AI for Science Workshop},
year={2024},
url={https://openreview.net/forum?id=YcZm8vteqE}
}
```



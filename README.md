
# CAAD 2018 Solutions

These are the solutions from team ysharma1126 for the
[Competition on Adversarial Attacks and Defenses 2018](http://hof.geekpwn.org/caad/en/).

Placed **1st**, **1st**, and **3rd** in non-targeted attack, targeted attack, and defense competitions. See our [Tech Doc](https://arxiv.org/abs/1810.01268) for further detail on the solutions.

### Citation
If you use the presented methods in your research, please consider citing

    @article{sharma2018caad,
      title={CAAD 2018: Generating Transferable Adversarial Examples},
      author={Sharma, Yash and Le, Tien-Dung and Alzantot, Moustafa},
      journal={arXiv preprint arXiv:1810.01268},
      year={2018}
    }

### Data

Download checkpoints for all attacks and defenses, as well as the data:

```bash
./download_data.sh
```

## Non-Targeted Attack
Placed **1st** in competition. Scaled the perturbation generated by a trained adversarial transformation network (ATN), allowing the optimization to be performed offline, thereby avoiding being limited by the stringent time constraint.

## Targeted Attack
Placed **1st** in competition. Couple spatial gradient smoothing with optimizing through randomization to significantly improve the transferability of MIM, while maintaining white-box success under the time constraint.  

## Defense
2_MSB placed **3rd** in competition. Add mild bernoulli noise to the input, pass the 2 MSB of the noisy example to an ensemble of adversarially trained classifiers for prediction, and patch the defense to improve its performance against our submitted non-targeted attack. 

Dropout amplifies the random resizing and padding defense with random dropout, using smoothing to fill in the selected dropped out values. This addition improves performance against strong attacks, but hurts clean accuracy.

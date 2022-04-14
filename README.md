## Disentangled estimation of reverberation parameters using temporal convolutional networks


### Introduction

This is the repository of the RevNet project. Our original paper can be found [here](https://aes.org/link).

A data-driven approach for estimating room reverberation parameters is proposed and evaluated. A dataset containing heterogeneous audio (speech, noise, etc.) is formed. Several types of artificial reverberation are selected and applied to the measurement signals, using different parameterizations (e.g. reverberation time, signal-to-reverberation ratio etc.). The dataset is used to train deep convolutional neural networks to estimate such parameters. Effects of dataset size, sample duration, speaker, and room dependence are also reported. 

![RevNet](model.png)


### Dependencies

You can install the requirements either to your virtualenv or the system via pip with:

```
pip install -r requirements.txt
```

### Data
To run the experiment:

1. Download the train-clean-100 subset of the [LibriSpeech ASR corpus](https://www.openslr.org/12) 
2. Download [OrilRiver Reverb Plugin](https://www.kvraudio.com/product/orilriver-by-denis-tihanov) by Denis Tihanov, [TAL-Reverb-4](https://tal-software.com/products/tal-reverb-4), or any other reverb VST3 (you will need to write a simple loader for custom VSTs).
3. Download [UrbanSound8k](https://zenodo.org/record/1203745#.YiZg1C8Rpqs) and resample all audio files from 22.05kHz to 16kHz using the following function:
```
NoiseReal().split_resample_urban(duration=10, fs=16000)
```
5. Replace the folder root directories in the Reverb.py file
```
LIBRISPEECH_PATH = 'your/folder/LibriSpeech'
NOISE_PATH = 'the/root/folder/of/' # UrbanSound8k
```
6. Run Reverb.py file to reproduce our results. This with save the following figures in your project folder.


### Training

Once you have the datasets prepared you can simply run the training process in a python console using:

```
sample_len = 6 # seconds
trainer = Trainer(sample_len=sample_len)
trainer.generate_sim_dataset()
trainer.train(epochs=100)
```
By default this will train on the available CUDA-capable GPU in your system.

**L1 loss**

![Loss TAL-Reverb-4](https://github.com/ithoidis/Reverb-parameter-estimation-using-temporal-CNNs/blob/main/results_TAL-Reverb-4/plots/train_history_param_tal.png)

![Loss OrilRiver](https://github.com/ithoidis/Reverb-parameter-estimation-using-temporal-CNNs/blob/main/results_OrilRiver/plots/train_history_param_oril.png)

### Load model and predict
```
trainer.load_model('models/RevNetGLU.pt)
y = trainer.model(x)
```
where x is a torch tensor of a reverberated speech signal.

#### Evaluate model

```
trainer.test()
trainer.test_in_noisy()
trainer.export_results()
```

### Authors

* **Iordanis Thoidis**, Aristotle University of Thessaloniki, Department of Electrical and Computer Engineering, Laboratory of Electroacoustics and TV Systems
* **Nikos Vryzas**, Aristotle University of Thessaloniki, Department of Journalism and Mass Media, Laboratory of Electronic Media
* **Lazaros Vrysis**, Aristotle University of Thessaloniki, Department of Journalism and Mass Media, Laboratory of Electronic Media
* **Rigas Kotsakis**, Aristotle University of Thessaloniki, Department of Journalism and Mass Media, Laboratory of Electronic Media
* **George Kalliris**, Aristotle University of Thessaloniki, Department of Journalism and Mass Media, Laboratory of Electronic Media
* **Charalampos Dimoulas**, Aristotle University of Thessaloniki, Department of Journalism and Mass Media, Laboratory of Electronic Media
 
### Reference

```
@article{thoidis2022reverb,
  title={Disentangled estimation of reverberation parameters using temporal convolutional networks},
  author={Thoidis, Iordanis and Vryzas, Nikolaos and Vrysis, Lazaros and Kotsakis, Rigas and Kalliris, George and Dimoulas, Charalampos},
  journal={AES 152nd Convention Papers},
  year={2022}
}
```

### Notes

* If using this code, parts of it, or developments from it, please cite the above reference.
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
* We assume no responsibility regarding the provided code.

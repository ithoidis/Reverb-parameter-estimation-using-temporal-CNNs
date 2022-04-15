## Disentangled estimation of reverberation parameters using temporal convolutional networks

This is the repository of the ReverbNet project, where we use deep learning to estimate the room reverberation parameters from speech utterances. Our original paper can be found [here](https://aes.org/link).

### Abstact
A data-driven approach for estimating room reverberation parameters is proposed and evaluated.In this study, we propose ReverbNet, an end-to-end deep learning-based system that processes raw audio waveforms to non-intrusively estimate multiple reverberation parameters from a single speech utterance. We employ a temporal convolutional network to map reverberated speech signals to a latent embedding space, and then predict the effect parameters that have been applied to the anechoic signals using a multi-branch head network. The proposed approach is evaluated using simulated room reverberation by two popular  effect processors. We show that the proposed approach can accurately estimate multiple reverberation parameters from speech signals and can generalise to unseen speakers and diverse simulated environments. The results also indicate that the use of multiple branches disentangles the embedding space from misalignments between input features and subtasks, and has a beneficial effect on the estimation of individual reverberation parameters.

![RevNet](model.png)


### Dependencies

You can install the requirements either to your virtualenv or the system via pip with:

```
pip install -r requirements.txt
```

### Data
To run the experiment:

1. Download the train-clean-100 and test-clean subsets of the [LibriSpeech ASR corpus](https://www.openslr.org/12) 
2. Download [OrilRiver Reverb Plugin](https://www.kvraudio.com/product/orilriver-by-denis-tihanov) by Denis Tihanov, [TAL-Reverb-4](https://tal-software.com/products/tal-reverb-4), or any other reverb VST3 (you will need to write a simple loader for custom VSTs).
3. To reproduce also the effects of noise on reverberation parameter estimation, please also download the [UrbanSound8k](https://zenodo.org/record/1203745#.YiZg1C8Rpqs) dataset and resample all audio files from 22.05kHz to 16kHz using the following function:
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

Once you have prepared the datasets, simply run the training process in a python console using:

```
sample_len = 6 # seconds
trainer = Trainer(sample_len=sample_len)
trainer.generate_sim_dataset()
trainer.train(epochs=100)
```
By default this will train the ReverbNet model on the available CUDA-capable GPU in your system.

**L1 loss**

![Loss TAL-Reverb-4](https://github.com/ithoidis/Reverb-parameter-estimation-using-temporal-CNNs/blob/main/results_TAL-Reverb-4/plots/train_history_param_tal.png)

![Loss OrilRiver](https://github.com/ithoidis/Reverb-parameter-estimation-using-temporal-CNNs/blob/main/results_OrilRiver/plots/train_history_param_oril.png)


### Evaluation
The following function perform the evaluation stages of the model (Evaluate using the Librispeech test set, Evaluate in noisy conditions using the UrbanSound8k dataset) and exports the figures in a new folder.
```
trainer.test()
trainer.test_in_noisy()
trainer.export_results()
```

### Inference
Let x be a torch tensor containing a reverberated speech sample waveform (shape: (1, time_samples), Float tensor). Type the following commands to get the model inference

```
trainer = Trainer(sample_len=6)
trainer.load_model('models/ReverbNet_OrilRiver.pt)
y = trainer.model(x)
```
where y (shape: (1, N_parameters)) is a vector containing the estimated reverberation parameter values corresponding to ['wet_db', 'reverb_db', 'decay_time_sec', 'room_size'] (in the default configuration).


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

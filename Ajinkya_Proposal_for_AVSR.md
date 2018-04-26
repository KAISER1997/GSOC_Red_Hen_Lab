# Audio and Visual Speech Recognition (AVSR) using Deep Learning

## Contact Information

**Name :** Ajinkya Takawale<br/>
**University :** [Indian Institute of Technology, Kharagpur](http://www.iitkgp.ac.in/)<br/>
**Email-id :** [ajinkya.takawale@iitkgp.ac.in](mailto:ajinkya.takawale@iitkgp.ac.in) ,      [ajinkya.takawale97@gmail.com](mailto:ajinkya.takawale97@gmail.com)<br/>
**GitHub username :** [ajinkyaT](https://github.com/ajinkyaT)<br/>
**Time-zone :** UTC + 5:30

## Abstract

Now, it is common for news videos to incorporate both auditory and visual modalities. Developing a multi-modal Speech to Text model seems very tempting for these datasets. The next goal is to develop a multi-modal Speech to Text system (AVSR) by extracting visual modalities and concatenating them to the previous inputs.

This project would extend already implemented previous GSOC candidate Divesh Pandey's speech recognition project.[\[1\]](https://github.com/pandeydivesh15/AVSR-Deep-Speech) The implemented work was based on the paper Deep Speech [\[2\]](https://arxiv.org/abs/1412.5567) but the authors of the original paper have proposed the second version of the same, Deep Speech 2 [\[3\]](https://arxiv.org/abs/1512.02595) which is completely End-to-End solution with no hand engineered features and also more efficient. I will primarily focus on using methods that are more accurate and robust in terms of computing power.

## Technical Details

### Overview of previous work

Following the approach in this paper, 'Deep complementary bottleneck features for visual speech recognition,'[\[4\]](https://ibug.doc.ic.ac.uk/media/uploads/documents/petridispantic_icassp2016.pdf) Restricted Boltzmann Machines and Autoencoders were used in the project. RBMs were responsible for finding proper initial weights for Autoencoder. Later, this autoencoder when fully trained was responsible for finding bottleneck visual features. Taken from the paper, model architecture looks like this, 

![](https://pandeydiveshnotgeek.files.wordpress.com/2017/07/auto_enc1.png)

Divesh also implemented speaker tracking. In news videos, there are times when we see two or more people in a single frame. We need to detect the proper speaker in these cases and perform lip reading accordingly. He was able to implement this to some extent and finally trained another model. The result obtained is shown below,

![](https://lh3.googleusercontent.com/PJksBILUjC_c-6ZrUA9wnIK-UXpmYwMyI51Xd-JKmr67YopfKqPMXOk0k46HitdBiTs698z1_tK2PE0=w1600-h794)


 
### Proposed Extensions

#### AVSR:

Conventional automatic speech recognition (ASR) typically performs multi-level pattern recognition tasks that map the acoustic speech waveform into a hierarchy of speech units. But, it is widely known that information loss in the earlier stage can propagate through the later stages. After the resurgence of deep learning, interest has emerged in the possibility of developing a purely end-to-end ASR system from the raw waveform to the transcription without any predefined alignments and hand-engineered models. However, the successful attempts in end-to-end architecture still used spectral-based features, while the successful attempts in using raw waveform were still based on the hybrid deep neural network - Hidden Markov model (DNN-HMM) framework. The paper, 'Attention-based Wav2Text with Feature Transfer Learning
,' [\[5\]](https://arxiv.org/pdf/1709.07814.pdf) constructed the first end-to-end attention-based encoder-decoder model to process directly from raw speech waveform to the text transcription. 

This recent paper, 'State-of-the-art Speech Recognition With Sequence-to-Sequence Models,' [\[6\]](https://arxiv.org/abs/1712.01769) introduces a sequence-to-sequence model with multi-head attention architecture, which offers improvements over the commonly-used single-head attention in sequence-to-sequence models.

I would therefore like to improve upon Divesh's work to incorporate end-to-end training along with attention based sequence-to-sequence model.

Although Google's Deepmind introduced WaveNet [\[7\]](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) for text to speech synthesize, but this implementation [\[8\]](https://github.com/buriburisuri/speech-to-text-wavenet) successfully showed the reverse implementation of speech to text. The model architecture is shown below.

![](https://raw.githubusercontent.com/buriburisuri/speech-to-text-wavenet/master/png/architecture.png)

I will try to incorporate visual modal into the above implementations to make them multi-modal system following approaches mentioned in the papers, 'Auxiliary Multimodal LSTM for Audio-visual Speech
Recognition and Lipreading' [\[9\]](https://arxiv.org/abs/1701.04224), and 'Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks
' [\[10\]](https://arxiv.org/abs/1703.10893v6)

**LipNet** [\[11\]](https://arxiv.org/abs/1611.01599is) the first end-to-end sentence-level lipreading model that simultaneously learns spatiotemporal visual features and a sequence model. The model maps a variable-length sequence of video frames to text, making use of spatiotemporal convolutions, a recurrent network, and the connectionist temporal classification loss, trained entirely end-to-end. The model architecture is shown below,

![](https://image.slidesharecdn.com/dlsl2017d4l4multimodaldeeplearning-170127184643/95/multimodal-deep-learning-d4l4-deep-learning-for-speech-and-language-upc-2017-51-638.jpg)

Keras implementation of LipNet is found here [\[12\]](https://github.com/rizkiarm/LipNet). Example of the above model is shown below,

![](https://raw.githubusercontent.com/rizkiarm/LipNet/master/assets/lipreading.gif)

It is therefore tempting to benchmark various approaches mentioned so far for the task of speech recognition.

#### Speaker Tracking:

As shown previously Divesh's work only highlights the lip region of the speaker. We could further improve this by predicting a bounding box around the faces in the video and predicting a right speaker as shown below. 

![](https://herohuyongtao.github.io/publications/speaker-naming/teaser.png)

Image is taken from the paper, 'Deep Multimodal Speaker Naming', [\[13\]](https://arxiv.org/abs/1507.04831)

Matlab implementation for the above is available here.[\[14\]](https://bitbucket.org/herohuyongtao/mm15-speaker-naming)

The author of the original paper further refined his approach by using multimodal LSTM which proved to be robust to both image degradation and distractors. Below is the comparison.

![](https://herohuyongtao.github.io/publications/look-listen-learn/teaser.png)

Figure: (a) Face sequence with different kinds of degradations and variations. Using the previous CNN methods cannot recognize the speakers correctly. In contrast, the speakers can be successfully recognized by our LSTM in both single-modal and multimodal settings. (b) Our multimodal LSTM is robust to both image degradation and distractors. Yellow bounding boxes are the speakers. Red bounding boxes are the non-speakers, the distractors.

I will therefore try to replicate above approaches and try to improve upon incorporating attention mechanism for the LSTM one.


#### Web App Demo:

Upon successful completion of the above tasks, we can make a Wep App demo using Flask Python library.

Process for the same is explained here.[\[15\]](https://bitbucket.org/herohuyongtao/mm15-speaker-naming)

This implementation shows the demo for simple image classification. [\[16\]](https://github.com/mtobeiyf/keras-flask-deploy-webapp)


**Keras** is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. I will therefore primarily try to build models using Keras, given that I have some past experience with it. TensorFlow and PyTorch will be used if attempts to implement in Keras fails owing to the flexibility of these frameworks.


## Schedule of Deliverables

Tasks for this project will be divided into two milestones

#### Milestone 1: Getting state-of-the art or comparable result

- Obtain relevant data and develop pre-processing pipelines.
- Find and study implementations available online of the papers mentioned above. If not, begin implementing from scratch.
- Compare results from different datasets.

#### Milestone 2: Building Web API and documentation

- Build demo Web API for the models implemented given an user option to upload any sample input.
- Document all the training and steps giving successful reproducibility a top priority. 

I hope to achieve Milestone 1 before midterm evaluation II. My time commitment will be 35-40 hours per week between April 28 - July 14 and 30-35 hours per week between July 15 - August 14. Below is task divided by dates.

#### April 23 - May 14 (Community Bonding Period):

- I will be having my end semester exams till April 27th. So my time will be limited in that period.
- I have primarily used Keras and TensorFlow for all the Deep Learning tasks and haven't yet used Pytorch, so I will try to get my hands dirty on PyTorch.
- I will also start maintaining a blog describing my weekly work summary.
- Although most of the datasets required for the task have been available from Red Hen Lab, if needed I will also try to gather other data sources.

#### Week I ( May 14 - May 20)
- Build pre-processing pipelines for audio only speech to text.
- Build pre-processing pipelines for multi-modal speech to text.
- Start 'State-of-the-art Speech Recognition With Sequence-to-Sequence Models' [\[6\]](https://arxiv.org/abs/1712.01769)


#### Week II ( May 21 - May 27)

- Continue working on above paper.
- Start with Convolutional variants of Siamese Network mentioned in technical details.
- Start implementing WaveNet [\[7\]](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

#### Week III ( May 28 - June 3)

- Try to finish previous implementations.
- Incorporate visual modal into above methods.
- Start 'Auxiliary Multimodal LSTM for Audio-visual Speech
Recognition and Lipreading' [\[9\]](https://arxiv.org/abs/1701.04224), and 'Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks
' [\[10\]](https://arxiv.org/abs/1703.10893v6)
- Keep maintaining blog!

#### Week IV (June 4 - June 10)

- Refine and debug everything so far.
- By now I hope to achieve reasonably good result for multi-modal AVSR.
- Summarize everything done so far on the blog.

#### Mid Term Evaluation I (June 11 - June 15)

Work on feedback/review (if any)

#### Week V ( June 18 - June 24)

- Finish implementing LipNet [\[11\]](https://arxiv.org/abs/1611.01599is). Shouldn't take much time since its implementation in Keras already exists.

#### Week VI ( June 25 - July 1) 

- Start working on Speaker Tracking. 
- Implement 'Deep Multimodal Speaker Naming', [\[13\]](https://arxiv.org/abs/1507.04831) which is based on CNN
- Keep maintaining blog!

#### Week VII ( July 2 - July 8)

- Finish CNN based approach and try its improvised LSTM based approach.
- Try to finish Speaker Tracking,

#### Mid Term Evaluation II (July 9 - July 13)

Work on feedback/ review, if any.
Refine/ debug

#### Week VIII ( July 16 - July 22)

- Start working on building demo Web App for everything done so far.
- Simultaneously start working on documentation.

#### Week IX ( July 23 - July 29)

- Continue with Web App and documentation.


#### Week X ( July 30 - Aug 5)
- Towards finalizing on Web App and documentation.
- Refine/ debug.
- Ask for feedback.

#### Submit final work ( Aug 6 - Aug 14)

- Finalize codes and documents for final submission.
- Work on feedback/review, if any.
- Wrap up everything.
 

## More About Me

I am a third year undergraduate at IIT Kharagpur majoring in Mechanical Engineering. Towards the end of my freshman year I discovered this wonderful field of Machine Learning and its applications. Intrigued by this field, I completed multiple certified MOOC's on the topics such as Data Science and Machine Learning. After exploring the field for quite some time then and being fascinated by the idea of automated future, I decided to dedicate my near future to the field of Machine Learning applications. 

So far I have done three internships in the field of Data Science and Machine Learning with the recent one being to related to finding potential use cases of Deep Learning applications in the medical images. Two of my internships were related to NLP, one in sentiment analysis of reviews and the other one developing query based content retrieval system using tf-idf similarity. Apart from that I regularly take part in Machine Learning competitions and Hackathons and had also won in one of them wherein I got a chance to attend DataHack Summit 2017[\[17\]](https://www.analyticsvidhya.com/datahacksummit/), India's largest conference on Artificial Intelligence and Machine learning. 

Through my previous experiences, I have gained  proficiency in common Python Libraries associated with data processing and Machine Learning tasks like **Numpy, Pandas, Matplotlib, scikit-learn, Keras and TensorFlow**. Below is the list of some of my past work done relevant for this project.

- Deep Learning for Medical Images[\[18\]](https://github.com/ajinkyaT/RoroData_internship)
- Following programming assignments completed in the Sequence Models Course[\[19\]](https://github.com/ajinkyaT/Coursera_Sequence_Models) 
  * Character-Level Language Modeling
  * Music generation with LSTM
  * Neural Machine Translation with Attention
  * Trigger word detection from speech data
  
- Following programming assignments completed in the Convolutional Neural Networks[\[20\]](https://github.com/ajinkyaT/Coursera_Convolutional_Neural_Network) 
  * Facial expression image classifier using Residual Network
  * Car detection using YOLOv2
  * Art generation using Neural Style Transfer
  * Face recognition using triplet loss function
  
- Simple Twitter Sentiment Analysis using TextBlob Python library[\[21\]](https://github.com/ajinkyaT/twitter_sentiment_analysis)
- Web crawler written for scraping amazon.in data[\[22\]](https://github.com/ajinkyaT/ori/blob/master/ama.py)

I am also attaching my [CV](https://drive.google.com/open?id=1nAjj8Z86lR7olGSxz9nWdHGBJH4BgLXo) for a more detailed explanation of my previous experiences.


## Why this project?

I am also planning to obtain Master's in Data Science after getting relevant work experience in this field. A project like this will add a lot of value to my application for the program. If possible, I would also like to contribute to Red Hen Lab after GSOC period unofficially as a side project. 

## Acknowledgement 

Michael Pacchioli . Thank you for giving valuable feedback and replying to my doubts!

## Appendix

Below is a list of all the web articles, GitHub repositories and research papers mentioned in the proposal.

1. https://github.com/pandeydivesh15/AVSR-Deep-Speech
2. https://arxiv.org/abs/1412.5567
3. https://arxiv.org/abs/1512.02595
4. https://ibug.doc.ic.ac.uk/media/uploads/documents/petridispantic_icassp2016.pdf
5. https://arxiv.org/pdf/1709.07814.pdf
6. https://arxiv.org/abs/1712.01769
7. https://deepmind.com/blog/wavenet-generative-model-raw-audio/
8. https://github.com/buriburisuri/speech-to-text-wavenet
9. https://arxiv.org/abs/1701.04224
10. https://arxiv.org/abs/1703.10893v6
11. https://arxiv.org/abs/1611.01599is
12. https://github.com/rizkiarm/LipNet
13. https://arxiv.org/abs/1507.04831
14. https://bitbucket.org/herohuyongtao/mm15-speaker-naming
15. https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
16. https://github.com/mtobeiyf/keras-flask-deploy-webapp
17. https://www.analyticsvidhya.com/datahacksummit/
18. https://github.com/ajinkyaT/RoroData_internship
19. https://github.com/ajinkyaT/Coursera_Sequence_Models
20. https://github.com/ajinkyaT/Coursera_Convolutional_Neural_Network
21. https://github.com/ajinkyaT/twitter_sentiment_analysis
22. https://github.com/ajinkyaT/ori/blob/master/ama.py

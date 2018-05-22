#  Session 3 - 음악과 오디오 분류 분야의 최신 기술 동향 

#### :: *이종필*

------

## 강의 내용 정리

**Music Information retrieval (음악정보검색)** 은 크게 세가지 로 나뉜다. 

- Factual information (artist, Composer, years … etc)
- Score-lecel Information (악보, melody, rhythm, chords)
- Semantic Information (Genre, , mood, instrument, text description)

(이 강연은 Semantic information (Genre, mood, instrument, text description)에 중점을 두고 있었다.)

### From Feature Engineering to End-to-End  Feature Learning 

* **Feature Engineering** 

  MFCC, hand-engineerd audio features

* **Feature Learning** 

  - Low level feature learning 
  - Deep convolutional Neural Networks
  - End-to-End feature learning 

#### 시간의 순서대로 모듈 구조의 변화 (1 $\rightarrow$ 4)

1. Spectrogram  $\rightarrow$ Linear to Mel $\rightarrow$ Magnitude Compression $\rightarrow$ Discrete Cosine transfer $\rightarrow$ Mean and Variation 
2. Spectrogram $\rightarrow$ Linear to Mel, CQT $\rightarrow$ Magnitude Compression $\rightarrow$ Affine Transform $\rightarrow$ Non-linearity $\rightarrow \cdots $ Pooling  $\rightarrow$ Classifier  **( Low level feature learning )**
3. Spectrogram $\rightarrow$ Linear to Mel CQT $\rightarrow$ Magnitude Compression  $\rightarrow$  Affine Transform $\rightarrow$ Non-linearity $\rightarrow $ Pooling  $\rightarrow \cdots$ Classifier **(CNN)**
4. **Raw Waveform** $\rightarrow$ Affine Transform $\rightarrow$ Non-linearity $\rightarrow $ Pooling $\rightarrow$ Affine Transform $\rightarrow$ Non-linearity $\rightarrow $ Pooling  $\cdots$ Classifier (End to End)



#### 1. Spectrogram  $\rightarrow$ Linear to Mel $\rightarrow$ Magnitude Compression $\rightarrow$ Discrete Cosine transfer $\rightarrow$ Mean and Variation  

* Linear-to-Mel : 달팽이관 흉내내기, 
* Magnitude Compression : 낮은 것은 높게, 높은 것은 낮게 (Contrast와 비슷한 기능)
* 모든 모듈이 각각이 모두 독립적이다. 

**Spectrogram**

![](http://www.youngkorean.com/waveform/2634to_stand_against.gif)

- FFT : Fast Fourier Transformation , 주파수 특성을 분석하는행위 (주파수 정보의 손실이 크다고 한다. )

**MFCC (Mel scale spectrogram)**

- **Scaled spectrogram = Mapping matrix $\times$ spectrogram**

**Magnitude Compression** (Amplitude Normalization)

![](https://librosa.github.io/librosa/_images/librosa-core-logamplitude-1.png)

**Mean and Variation**

- Variable length audio -> clip level representation 
- 분류기 - (GMM, SVM … etc)



#### 2. Low level feature learning 

* 위에서 Affine transform 이 추가 됨 

  Random sampling(여러개의 프레임을 샘플링) -> Unsupervised learning -> classifier 



#### 3. Audio CNN Models before End2end 

- 1D / 2D Convolution filter 혼용해서 사용하지만, 1D에 더 이점이 있다고 한다. 
- 심플한 구조의 CNN을 사용 , Frame-level Spectrogram 의 인풋을 사용 (이후 Raw data를 그대로 처리하는 모델로 발전 ) 

#### 4. End to End Model  

**기본적인 CNN을 사용**

- AlexNet , ZFnet, VGGNet, ResNet 같은 이미지에서 큰 성과 가 있었던 구조를 차용해서 사용. 

- 그 중 가장 좋았던 것

  10 layers 이상의 가장 작은 샘플 단위의 필터와 3 ~ 5 초정도로 서브샘플링된 오디오를 인풋으로 하였을 때이다. 

  ​

**Advanced Models** 

- Convolutional Recurrent Neural Networks 

  ![3-Figure1-1](/Users/MAC/Downloads/3-Figure1-1.png)

- Residual Networks and Squeeze and Excitation Networks 

  ![2-Figure1-1](/Users/MAC/Downloads/2-Figure1-1.png)

  - SEblock 이 성능을 많이 끌여올렸다. 
  - ReSE-2-multi sample CNN이 가장 성능이 좋다. 

- Pair-wise Learning 

  - Artist를 예측하는 문제로 바꿔보니, 500명 정도 사용해서 장르나 모드 예측과 비슷한 성능이 나옴 

    ![](https://image.slidesharecdn.com/music-data-start-to-end-171020150826/95/music-data-start-to-end-37-638.jpg?cb=1508512137)




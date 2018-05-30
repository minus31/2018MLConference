#  Session  - All about NMT 

#### :: *김기현*

## <u>강의 내용 정리</u>

### 1. **NMT Basics**

#### 기계번역의 Objective function ;

##### $\hat{e} = argmaxP_{\text{f}\rightarrow e}(e | \text{f})$

:: e = English, $f$ = French  ; French $\rightarrow$ English translation

MT(Machine Translation)은 Rule based의 전통적인 번역모델 부터 SMT(통계기반 번역)을 거치고, 2014년도부터 NMT( Neural Machine Translation )가 나오며 그 품질에 큰 성장 곡선을 그렸다. 

#### NMT가 잘되는 3가지 이유

- End2End 모델이기 때문에, 기존 기계번역이 필요로 했던 추가적인 작업이 필요없이, 처음 부터 끝까지 구현하는 것이 쉬워졌고, 정확도도 높아졌다. 
- Better language Model ; 한 번에 처리할 수 있는 단어의 수가 증가하였고, 더 나은 lingustic comprehensive를 갖추었다. 
- Great Context Embedding ; 단어 그 자체 뿐아니라 그 단어가 쓰인 문맥에 대한 정보도 embedding과정에 포함 시킬 수 있다. 

#### Sequence to Sequence

Seq2Seq는 도메인간의 이동을 위해 스스로를 Sequencial data로 압축하는 것이다. 

Encoder, Decoder, Generator의 구조로 이루어져있다. 

##### $\theta^* = argmaxP_\theta(Y|X) \ where \  X = {x_1, x_2, \cdots, x_n}, Y = {y_1,  y_2, \cdots, y_m}$

###### Encoder 

Encoder의 구조를 먼저 수식으로 표현

$h_t^{src} =RNN_{enc}(emb_{src}(x_t, h_{t-1}^{src})$

$H^{src} = [h_1^{src}, h_2^{src}, \cdot,  h_n^{src}]$

$\downarrow$

$H^{src} = RNN_{enc}(emb_{src}(X), h_0^{src})$

말하자면, 가변길이의 Input을 고정된 길이의 'Context vector' 로 만드는 과정이다. 

###### Decoder - more important 

Decoder 는 Conditional RNNLM(Recurrent Neural Network Language Model)의 일종이라고 이해하면 된다. 

:: 기본적으로 Language model 은 n개의 단어가 들어가서 다음 단어를 예측 하는 것이다. 

* RNNLM

$P(w_1,w_2, \cdots , w_k) = \prod_{i=1}^kP(w_i|w_{<i})$

i 시점의 word의 확률은  그 전 시점 word의 조건에서의 i시점의 word의 확률의 product이다. 

* Conditional RNNLM

  $P_\theta(Y|X) = \prod_{t=1}^mP_\theta(y_t|X, y<t)$

**Decoder**의 수식 표현 

##### $h_t^{tgt} = RNN_{dec}(emb_{tgt}(y_{t-1}), h_{t-1}^tgt)  where h_0^{tgt} = h_n^{src}andy_0 = BOS )$

:: BOS = Beginning of Sentence 

​					$\downarrow$

##### $H_t^{tgt} = RNN_{dec}(emb_{tgt}([BOS;Y[: -1]]), h_n^{src})$



$\rightarrow$ BOS를 Decoder가 던지면서 그 다음 단어들을 예측해간다. 

###### Generator 

$\hat{y}_t = softmax(linear_{hs\rightarrow |V_{tgt}|}(h_t^{tgt})) and \hat{y}_m = EOS  $

where hs is hidden size of RNN, and $|V_{tgt}|$ is size of output vocabulary.

::EOS = End of Sentence 

> Where to use;

> - NMT, Chatbot, Summarization, Other NLP, Automatic Speech Recognition, lip Reading, Image Captioning, etc

###### 한계 
    1. Memorization 
    2. Lack of Structural Information -> 사실 실제적으로는 무시해도 성적은 잘나오더라
    3. Domain Expantsion - 특히 Chatbot 구현 시에 잘 안되는 경우가 많다.
-------------

### 2. **Intermediate NMT**

#### What is Attention ?

**{ 쉽게 생각하면, IdLookupTable, Dictionary이다. }** ex)  >> {'dog':1, 'computer': 2, 'cat':3 ...}

<CODE>

```
def key_value_func(query):
	weights = []
	
	for key in dic.keys():
		weights +== [is_same(key, query)]
	answer = 0 
	for weight, value in zip(weights, dic.values()):
		answer += weight * value
		
	return answer
-------------------------------------------------------
# dictianary의 values에는 n차원의 벡터가 들어가있고, query와 key 값이 vectordlrh, dict의 key 값과 value값이 같다면

word2vec('dog')
>> [0.1, 0.3, -0.11, ....]
word2vec('computer')
>>[0.13, 0.14, 0.39, ....]

dic = {word2vec('doc') : word2vec('doc'), word2vec('computer') : word2vec('computer')}
answer = key_value_func(word2vec(Query))
--------------------------------------------------------
def key_value_func(query):
	weights = []
	
	for key in dic.keys():
		weights += [how_similar(key, query)] #dot_product값을 채워준다. ( cosine similarity )
	weights = softmax(weights) # 모든 가중치를 구한 후 softmax 계산을 한다. 
	answer = 0 
	for weight, value in zip(weights, dic.values()):
		answer += weight * value
		
	return answer
```
</code>

---------


Key value function을 미분 가능하게 만들기 위해서, is_same(binary result)을 묻는 것이 how_similar(softmax)를 목적으로 한다. 그 확률이 가장 높은 것을 찾는 것이다(argmax).

여기서, dictianary의 values에는 n차원의 벡터가 들어가있고, query와 key 값이 vectordlrh, dict의 key 값과 value값이 같게 하면 **Attention**의 작동원리를 쉽게 이해 할 수 있을 것이다. 

:: Query = 현재 time step 의 decoder OUTPUT

:: keys = 각 time-step별 encoder OUTPUT

:: Values = 각 time step별 encoder OUTPUT (->key값과 같게 가정)

- 수식으로 표현

$ w = softmax(h_t^{tgtT}W H^{src}) \ \rightarrow \ LinearTransformation \ 한 \ 것을 \ softmax한다.  $  

$c = H^{src} \ w \ and \ c \ is \ a \ context \ vector$  

$\tilde{h_t}^{tgt} = tah(linear_{2hs\rightarrow hs}([h_t^{tgt};c ]))$

$\hat{y_t} = softmax(linear_{hs\rightarrow |V_tgt|}(\tilde{h_t^{tgt}}))$

$where \ hs \  is \ hidden \ size \ of RNN,\ and \ |V_tgt| \ is \ size \ of \ output \ vocabulary.$ 

여기서 linear transformation 하는 이유는 ; 선형적 조합을 통해 어느정도 다른 도메인끼리 문맥상 유사한 단어들끼리 1:1 매칭이된다고 가정하기 때문이다. 

**-> Attention을 함께 썼을 때, 그 때까지 있던 것들을 속도와 성능 면에서 앞섰다. **



####  Input Feeding

#####  -> 이부분은 argmax할 때 정보손실이 난다는 것을 보완하기 위해, concat(linear transformation)을 시킨 후 와 softmax취한 후, 두값을 각각 다음 임베딩에 인풋, 임베딩 전처리 레이어의 인풋으로 주면서 정보손실을 최소화 시켜서 Feeding한다는 내용을 언급을 하고 넘어갔습니다.  ( $\rightarrow$ 인풋피딩하는 방법론은 stacked RNN Seq2Seq의 모델 그대로 하는 것을 보입니다. ), InputFeeding의 단점으로 훈련시 매 타임스텝마다 따로따로 구해줘야 해서 시간이 많이 걸린다는 것이였지만, 사실 inference할 때 만 괜찮다면, Training시에는 시간이 걸려도 문제가 크지 않다.  

> 방법에 따른 성능 변화 ( 꾸준히 개선됨 )

|               NMT system                | perplexity | BLEU |
| :-------------------------------------: | :--------: | :--: |
|                  base                   |    10.6    | 11.3 |
|             base + reverse              |    9.9     | 12.6 |
|        base + reverse + dropout         |    8.1     |  14  |
|  base + reverse + dropout + attention   |    7.3     | 16.8 |
| base + reverse + attention + feed input |    6.4     | 18.1 |

#### Search

Beam Search( greedy search를 하되 1개씩이 아니라 k개씩 greedy search 한다.  (수식은 넘어감)

$\hat{y_t}^k  = argmax_{k-th}\hat{Y_t}$ 

$\hat{Y_t} = f_\theta(X, y^1_{<t}) \cup \cdots \cup f_\theta(X, y^k_{<t})$

$X = \{ x_1, x_2, \cdots,x_n \}$

$\rightarrow$ **Beam search로 최적화를 했더니 성능이 올랐다.** 



#### Evaluation

#####Length Penelty 

이 페널티가 없으면, 모델은 짧은 문장만 고르게 된다. 

 $ \log{\tilde{P}}(\hat{Y}|X) = \log{P(\hat{Y}|X)} * penalty$

$penalty = \dfrac{(1+ length)^\alpha}{(1+\beta)^\alpha}$

$where \ \beta \ is \ hyper \ parameter \ of  \ minimum  \ length $

$\rightarrow$ Length penelty 는 hyper parameter 이지만  

##### Cross entropy and Perplexity 

:: language model 의 성능을 측정할 때 쓰는 loss function이다. 



$\log{(\sqrt[|Y|] {\dfrac{1}{\prod_{y \in Y}{P_\theta(y)}})}}$

$PPL(W) = P(w_1, w_2, \cdots, w_N)^{-\dfrac{1}{N}} = \sqrt[N]{\dfrac{1}{P(w_1, w_2, \cdots , w_N)}}$ 

$chainrule \rightarrow PPL(W) = \sqrt[N]{\prod{\dfrac{1}{P(w_i|w_1, w_i|w_2, \cdots , w_{i-1})}}}$

$PPL = \exp(Cross Entropy) \  그러나 \ 정확한 \ 지표는 \ 아니라고 \ 한다. $

평균적으로, 각 타임스텝별 후보가 N개 있다. 

사람의 눈으로는 더 나쁜 번역이 PPL로는 더 나은 번역으로 나올 때가 있다. 이를 보완 하기 위해서, BLEU도 같이 쓴다. 



##### BLEU

:: 실제 품질과 관계가 높고 

sementically 한 것이 더 잘 표현이 된다. n그램 당 히트수가 많으면 높아지고, 높을 수록 좋다 .

**그러나, Back Propogation이 불가능 하다.** 

$BLEU = brevity-penalty * \prod_{i=1}^{n}p_i^{w_i}$

$brevity-penalty = min(1, \dfrac{|prediction|}{|reference|})$

$where \ p_i \ is \ precision \ of  \ i-gram \ and \ w_i \ is \ weight$



#### Conclusion 

* Attention - finding a 'value' which has similar 'key' to 'Query'.

----------------

### 3.**Advanced Topic in NMT**

#### Zero shot Learning ( 한번도 보지 않은 데이터를 다룰 수 있는가 ?) 

Encoder 와 Decoder 에 다수의 언어를 넣고 훈련 시키고(multi domain), 훈련된 모델에서 zero shot translation의 성능을 평가한다. - 구현이 쉽지만 **성능은 그닥 좋지 않다.** 그러나, 정중한 번역 , 사투리 번역 등의 세부적인 domain의 번역이 가능하고 의료도메인 번역 법률 도메인 번역 등의 전문적인 번역이 가능할 것으로 기대된다. 

#### $\rightarrow$  zero shot translation과 일반 번역모델의 결과를 비교 하여주셨다. 여기서 BLEU가 얼마정도 증가하면 의미있는 증가인지 한번 언급 해주셨으면 했다. 

#### Language Model Ensemble (from Gengio`  LAP)

Shallow Fusion 에 비해 Deep Fusion쪽이 성능이 더 좋다. (두 구조의 차이에 대해서는 설명을 안해주셨다. 그러나 Seq2Seq에서 Decoder conditional language model 이기때문에, 구조는 decoder와 소이 하고 안에서 스텝간의 가중치 흐름에서 다음 스텝으로 가는 값이 몇가지 증가 했다. )



#### Back Translation 

-> Language model ensemble보다 구현 이 쉽다. -> 모델을 바꿀 필요없고, Data를 Augment시키는 방법론이다. 

쉽게 설명하여, 영 - 한 번역 을 목표로 할 때, 훈련 시, 한 - 영 데이터로 훈련시키다. 

#### Copied Translation

위의 것과 비스한 개념으로 모델을 바꾸지 않고 data를 augment시켜서 성능을 높이고자 하는 시도이다. 여기서는 앞뒤가 똑같은 데이터를 더 훈련시킨다. 

#### Fully Convolutional Seq2Seq 

업계의 주류는 RNN기반의 Seq2Seq 이지만, 페이스북에서 전부 Convolution으로 이루어져있는 Seq2Seq를 발표 하였다. 

Convolution filter 만 통과해도 어느정도 encoding이 된다는 것을 가정으로 만들어 졌다. -> 그러나 구현이 복잡하여 잘 안쓰인다. 

#### Transfomer - "Attention is all you need" 이란 논문을 통해 발표됨

$\rightarrow$ Attention을 한 층(layer)에서 여러층의 Attention을 적용한다. 

<수학적 표현을 빌려 이 개념을 설명> -강사님은 그냥 넘어가셨다. 

$Attention(Q, K, V) = softmax(\dfrac{QK^T}{\sqrt{d_k}})V$

$MultiHead(Q, K, V) = [head_1; head_2, \cdots, head_n]W^o$

$where head_i = Attention(QW_i^Q, KW_i^K, VW_I^V)$

$where W_i^Q \in \mathbf{R}^{d_model \times d_k}, W_i^K \in \mathbf{R}^{d_model \times d_k}, W_i^V \in \mathbf{R}^{d_model \times d_v} and W^O \in \mathbf{R}^{hd_v \times d_model}$

$d_k = d_v = d_model / h = 64$

$ h  = 8, d_model = 512$

###### Attention case by case

* self-attention at Encoder, Decoder(masking to prevent, should not access future time-step) and Attention at Decoder $\rightarrow$ 3 types of Attention, each perform within their region 

$\rightarrow$ this makes the model to get better BLEU score 

--------------------
### 4.**Productization of NMT**

#### Pipeline 

 Collecting corpus $\rightarrow$ cleansing corpus 

$\rightarrow$ Tokenizing ( Mecab등을(Pos Tagger) 이용하여 세분화 ; Subword segmentation : Byte Pair Encoding; 띄어쓰기가 있는 언어(ex. 한국어; mecab은 본디 일본어용) 는 Normalizing 해준다 ;  ) 

$\rightarrow$ Batchfy ( 단어 길이가 같은 것 끼리 묶어줌(padding 을 제외한 길이), OutofVacaburary 가 생기면 성능이 저하되기 때문에, 예를 들어 "학교 ^ 에" 로 따로 넣는 것이 OoV를 그나마 줄이게 된다. ) $\rightarrow$

 Training $\rightarrow$ Inference $\rightarrow$ Restore tokenization

#### 이 파트에서 다루어 졌던 개념들 

* Google`s NMT - 기계번역의 바이블 

* Residual Connection / Residual Network

* Bi-directional Encoder for First Layer - 시간 제약을 극복하기 위해, 첫번째 Layer에서만 양방향 학습이기때문에 parameter size가 준다. 

* WordPiece model (:: subword; 현재 NMT에는 기본적으로 subword 개념이 사용된다. ) == BPE

  > original - jet makers feud over seat width with big orders at stake 
  >
  > wordpieces - _J (단어의 시작) _(띄어쓰기)makers -fe ud _over_seat_width _with_ _big _orders _at _stake

* Quantization 

  > 양자화 훈련 - 계산량을 줄일 수 있다. 실제 모델이 저장되는 크기를 줄여 deploy 를 효율적으로 할 수 있다. 부가적으로 Regularization기능 (정확도를 다소 희생)

  $ Length Penalty and Coverage P, X) = \log{ P(Y|X)} /  lp(Y) + cp(X;Y)$

  $lp(Y)  = \dfrac{(5+|Y|)^\alpha}{(5+1)^\alpha}$

  $cp(X; Y) = \beta * \sum_{i=1}^|X|\log{(min(\sum_{j=1}^|Y|{P_{i,j}, 1.0}))}$

  $where \ p_{i,j} \ is \ the \ attention \ weight \ of \ the \ j-th \ target \ word \ y_i \ on \ the \ i-th \ source \ word \ x_i.$

* Training Procedure - SGD나 Adam 을 단독으로 쓰기 보다, Adam으로 하다가 SGD로 최적화한다. 

* Edinburgh`s NMT - Back Translation, Copied Translation, Subword 등의 근원지 이다. 

  ><Architecture> 
  >
  >* use GRU and residual connection - 4 layers for Encoder, 8 layers for Decoder 
  >* Hidden size = 1024
  >* Word vector dimension = 512 
  >* Use only Adam 
  >* Synthetic Data using Monolingual Data (parallel : copied : back = 1 : 1~2 : 1~2)
  >  * copied와 back data들이 과도하게 많아 지면 안된다. (parallel은 여기서 양방향 을 의미한다. )

* Ensemble 

  * Checkpoint ensemble - retrained model from certain epoch  (20 epoch까지 독립적인 훈련)
  * Indepedent ensemble - Train different model from the beinning 

$\rightarrow$ 여기 까지 적용하게 되면,  NMT의 성능이 괄목할 정도로 향상된다. (BLEU기준) - 얼마 정도가 얼만큼의 의미인지도 infer해주시길 바랐다. 

------

### 5.**Reinforcement Learning for NLP**

- Reasons of using RL - GAN 와 같이 할 수 있길 기대

  - MSE나 CrossEntropy 보다 더 복잡한 목적함수를 쓸 수 있게 한다. 

  - GAN이 NLP에서 사용될 수 없는 이유

    > Word is a symbol which is discrete 
    >
    > So, Pickiing a word is sampling(or argmax) which is stochastic process. 
    >
    > Stochastic Process cannot pass gradient, that means Back Propogation is not eligible

- **Policy Gradient** , to make back propogation possible

$\pi_\theta(a|s) = P_\theta(a|s) = P(a|s;\theta)$

$J_\theta = E_(\pi\theta) = \mathbf{E}_{\pi\theta}[r]$

$= \sum_{s \in S}{d(s)}\sum_{a \in A}{\pi_\theta(s, a)\mathbf{R_{s, a}}}$

$\theta_{t+1} = \theta_t + \alpha\nabla_\theta J(\theta)$

$\nabla_\theta \pi_\theta(s, a) =  \pi_\theta(s, a) \dfrac{\nabla_\theta \pi_t heta(s, a)}{\pi_\theta(s, a)}$

$= \pi_\theta(s, a) \nabla_\theta \log{\pi_\theta(s, a)}$

$\nabla_\theta J(\theta) = \sum_{s \in S}d(s)\sum{\pi_theta(s, a)\nabla_theta \log \pi_\theta(s, a) \mathbf{R}_{s, a}}$

$\nabla_\theta J(\theta) = \mathbf{E_{\pi\theta}}[\nabla_\theta \log \pi_\theta(a | s) r]$

$ \downarrow$

$\nabla_\theta J(\theta) = E_{\pi \theta}[\nabla_\theta \log{\pi_\theta(a | s)Q^{\pi\theta}(s, a)}]$



### 6. **Supervised / Unsupervised Learning for NMT with RL**

- Minumum Risk Training 

$\tilde{R}_\theta = \sum_{s=1}^s E_{(y|x^{(s)}; \theta, \alpha)[\nabla(j, y^{(s)})]}$

$= \sum_{s=1}^S\sum_{y \in S(x^{(s)})} Q(y|x^{(s)} ; \theta, \alpha)\nabla(y, y^{(s)})​$

$where \ S(x^{(s)}) \ is \ a \ sampled \ subset \ of \ the \ full \ search \ space \ y(x^{(s)}) \\ and \ Q(y|x^{(s)}; \theta, \alpha) \ is \ a \ distribution \ defiend \ on \ the\ subspace \ S(x^{(s)})$

$Q(y|x^{(s)}; \theta, \alpha) = \dfrac{P(y|x^{(s)}; \theta)^\alpha}{\sum_{\prime{y} \in S(x^{(s)})}P(\prime{y}|x^{(s)}; \theta)^\alpha}$

$\rightarrow$ Minimum Risk Training 으로 할 때  MLE방식 보다 BREU기준 2 ~ 3points 향상된다. 



* Policy Gradient for GNMT

$\mathbf{O}_{ML}(\theta) = \sum_{i=1}^{N} \log P_\theta(Y^{*(i)}| X^{(i)})$

$\mathbf{O}_{ML}(\theta) = \sum_{i=1}^{N}\sum_{Y \in \gamma} P_\theta(Y|X^{(i)}r(Y, Y^{* (i)}) \ \rightarrow letter \ part \ means \ Reward​$ 

$\mathbf{O}_{Mixed}(\theta) = \alpha * \mathbf{O}_{ML}(\theta) + \mathbf{O}_{RL}(\theta)$

$\rightarrow$ 소폭의 성능 상승 



* Unsupervised Learing for NMT with RL 

  > Cycle GAN ; Transition between multiple domains
  >
  > * Adversarial 구조를 NMT에도 적용 

* **Dual Learning for Machine Translation** (; Pre-training 된 모델을 이용한다. )

  $\theta_{AB} \leftarrow \theta_{AB} + \gamma \nabla\theta_{AB}\hat{E}[r]$

  $\theta_{BA} \leftarrow \theta_{BA} + \gamma \nabla\theta_{BA}\hat{E}[r]$

  $r = \alpha r_{AB} + (1 - \alpha)r_{BA}$

  $r_{AB} = LM_B(S_{mid})$

  $r_{BA} = \log{P(s|s_{mid}; \theta_{BA})}$

  ​

  $\nabla \theta_{AB}\hat{E}[r] = \dfrac{1}{K}\sum_{k=1}^{K}[r_k \nabla \theta_{AB} \log{P(s_{mid, k} | s ; \theta_{AB})}]$

  $\nabla \theta_{BA}\hat{E}[r] = \dfrac{1}{K}\sum_{k=1}^{K}[(1-\alpha) \nabla \theta_{BA} \log{P(s_{mid, k} | s ; \theta_{BA})}]$

  $\rightarrow$ this makes the model better by approximately 2 ~ 3 points, and overtakes the other models in a large scale of itself 

* **Unsupervised NMT** 

  * Denoising Auto-encoder 

    > $\mathcal{L}_{auto}(\theta_{enc}, \theta_{dec}, \mathcal{Z}, \mathcal{l}) = \mathbf{E_{x \sim \mathcal{D}_\mathcal{l}, \hat{x}\sim d(e(C(x), \mathcal{l}), \mathcal{l})}[\Delta(\hat{x}, x)]}$

  * Cross Domain Training (Translation)]

    > $y = M(x)$ 
    >
    > $\mathcal{L}_{cd}(\theta_{enc}, \theta_{dec}, \mathcal{Z}, \mathcal{l}_1, \mathcal{l}_2) = \mathbf{E}_{x \sim \mathcal{D}_{\mathcal{l}_1}, \hat{x} \sim d(e(C(y), \mathcal{l}_2), \mathcal{l}_1)}[\Delta(\hat{x}, x)]$

  * Adversarial Training 

    * Disciminator (어떤 언어인가) 

      > $\mathcal{L}_D(\theta_D|\theta, \mathcal{Z}) = -\mathbf{E}_{(x_i, \mathcal{l}_i)}[\log{p_D(\mathcal{l}_j |e(x_i, \mathcal{l}_i))}]$

    * Generator(Encoder)( making Hidden state ) 

      > $\mathcal{L}_{adv}(\theta_{enc}, \mathcal{Z}|\theta_D) = -\mathbf{E}_{(x_i, \mathcal{l}_i)}[\log{p_D}(\mathcal{l}_j|e(x_i, \mathcal{l}_i))] \ , \ where \  j = -(i - 1) $

  * **Final Objective**

    $\mathcal{L}(\theta_{enc}, \theta{dec}, \mathcal{Z}) = \lambda_{auto}[\mathcal{L}_{auto}(\theta_{enc}, \theta{dec}, \mathcal{Z}, \mathcal{l}_{src}) + \mathcal{L}_{auto}(\theta_{enc}, \theta_{dec}, \mathcal{Z}, \mathcal{l}_{tgt})] \\ + \lambda_{cd}[\mathcal{L}_{cd}(\theta_{enc}, \theta_{dec}, \mathcal{Z}, \mathcal{l}_{src}, \mathcal{l}_{tgt}) + \mathcal{L}_{cd} = (\theta_{enc}, \theta_{dec}, \mathcal{Z}, \mathcal{l}_{src}, \mathcal{l}_{tgt})] \\ + \lambda_{adv}\mathcal{L}_{adv}(\theta_{enc}, \mathcal{Z}|\theta_D) $

    ​

    ```python
    #Algorithm ; Unsupervised Training for Machine Translation 
    1: procedure Training(D_src, D_tgt, T)
    2: Infer bilingual dictionary using monoliingual data(conneau et al., 2017)
    3: M^(1) <- unsupervised word-by-word translation model using the inferred dictionary 
    4: for t = 1, T do
    	using M^(t), translate each monolingual dataset
        // discriminator training & model thrining as in eq. 4
        \theta_{discr} <- argmin Loss_D, \theta_enc, \theta_dec, \Z, <- argmin Loss
        M^(t+1) <- e^(t) 0 d^(t) // update MT model 
        end for f
        return M^(t+1)
    end precedure
    ```




### Conclusion

지금 까지,  Rule_based Machine Translation부터 Statistical Machine Translation를 지나, 이 들의 성과를 괄목하게 뛰어 넘은 Neural Machine Translation이론이 어떻게 그 성능을 개선해 왔는지를 포괄적으로 훓어 내려왔다. 지금까지 배운 것이 실제 연구에서 시도된 가장 발전된 형태의 NMT 방법론이다. 지금 업계는 **"이제 더이상 시도할 수 있는 지 모르겠다."** 는 반응이라고 하며, 첫번째 Session은 끝이 났다. 
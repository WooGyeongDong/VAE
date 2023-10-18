Auto-Encoding Variatioal Bayes Review
===========================
### Background
- 우리는 데이터 $X$와 비슷한 분포를 가지는 $p(x)$를 추정하고자 한다.

$X=\{x^{(i)}\}^N_{i=1}$ 

$x^{(i)}\sim p_{\theta}(x^{(i)}|z)\quad z\sim p_{\theta}(z)$

$x^{(i)}\sim p_{\theta}(x^{(i)})=\int p_{\theta}(x^{(i)}|z)p_{\theta}(z) dz$

가능도함수 $p_{\theta}(x^{(i)})$를 최대화 하는 $\theta$를 찾아야 하지만 일반적으로는 $\theta$에 대해 다루기 힘든 함수이다.



- 로그 가능도함수 $\log p_{\theta}(x^{(i)})$는 다음과 같이 쓸 수 있다.

$\log p_{\theta}(x^{(i)}) = D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})) + \mathcal{L}(\theta, \phi ; x^{(i)})$

$\log p_{\theta}(x^{(i)})  \geq  \mathcal{L}(\theta, \phi ; x^{(i)}) \quad\because D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})) \geq 0$

이제 우리는 가능도함수의 하한(lower bound of the likelihood)를 최대화 하는 것으로 문제를 바꿔 생각할 수 있다.



- 가능도함수의 하한은 다음과 같이 근사한다.

$\mathcal{L}(\theta, \phi ; x^{(i)}) = E_{q_{\phi}(z|x^{(i)})}[-\log q_{\phi}(z|x^{(i)}) +\log p_{\theta}(x^{(i)}, z)]$

$\simeq \frac{1}{L} \sum_{l=1} \log p_{\theta}(x^{(i)}, z^{(i,l)})-\log q_{\phi}(z^{(i,l)}|x^{(i)}) = \widetilde{\mathcal{L}}(\theta, \phi ; x^{(i)})$

$z^{(i,l)} = g_{\phi}(\epsilon^{(i,l)},x^{(i)}),\quad \epsilon^{(l)} \sim p(\epsilon)$


$p_\theta(z) = N(0,I)$, $p_\theta(x|z) = N(\mu_\theta(z), \sigma^2_\theta(z)I)$라고 가정하자 이때
$\mu_\phi(z)$, $\sigma^2_\phi(z)$는 $z$가 MLP를 통과한 것이다. 예를 들어,

$h = tanh(W_1z+b_1)$

$\mu = W_2h + b_2$

$\log \sigma^2 = W_3h + b_3$

비슷하게, $q_{\phi}(z|x) = N(\mu_\phi(x), \sigma^2_\phi(x)I)$라고 가정하고
$\mu_\phi(x)$, $\sigma^2_\phi(x)$는 $x$가 또 다른 MLP를 통과한 것이다.

$z^{(i,l)}=\mu_\phi(x_i) + \sigma^2_\phi(x_i)\odot \epsilon^{(l)}\quad\epsilon^{(l)} \sim N(0,I)$

정규분포 가정 시 KL-divergence term이 계산 가능하고 다음과 같은 추정량을 얻을 수 있다.

$\widetilde{\mathcal{L}}(\theta, \phi ; x^{(i)}) = -D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z)) + \frac{1}{L} \sum_{l=1}[\log p_{\theta}(x^{(i)}|z^{(i,l)})]$

### Data 학습
- 데이터는 MNIST데이터를 사용하였다.

![media_images_original_601_3b777ad78d25eca75335](https://github.com/WooGyeongDong/VAE/assets/143774643/ce76e235-ebbb-49cf-bbc9-d99b6fd6d8f3)

- 하이퍼파라미터는 논문을 참고하여 결정하였다.
'input_dim' : 28*28,
'hidden_dim' : 500,
'latent_dim' : 2,
'batch_size' : 100,
'epochs' : 100,
'lr' : 0.01,
'best_loss' : 10**9,
'patience_limit' : 3

### 결과

![W B Chart 2023  10  18  오후 7_14_41](https://github.com/WooGyeongDong/VAE/assets/143774643/73a910ca-1b43-4237-bfbf-65590b085001)

![W B Chart 2023  10  18  오후 7_15_12](https://github.com/WooGyeongDong/VAE/assets/143774643/e325db02-61e9-4e62-8d3b-fc132190250e)

- 위의 MNIST예시를 Input으로 생성한 결과
- 
![media_images_generate_100_db4ff0c9293a7e0799dd](https://github.com/WooGyeongDong/VAE/assets/143774643/3c5c2831-15a8-437c-8f0f-7d4f3208cbf6)

- Latent variable(z)에 따른 생성 결과
- 
![media_images_latent generate_100_3f3199a1788b5b61e25a](https://github.com/WooGyeongDong/VAE/assets/143774643/3a79a827-9b33-45d6-9b13-046a8f58bcc9)




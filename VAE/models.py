import torch
from torch import nn
import math

"""
Все тензоры в задании имеют тип данных float32.
"""

class AE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.encoder = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Генерирует код по объектам.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: Tensor - матрица размера n x d.
        """
        # ваш код здесь
        return self.encoder(x)

    def decode(self, z):
        """
        По матрице латентных представлений z возвращает матрицу объектов x.
        Вход: z, Tensor - матрица n x d латентных представлений.
        Возвращаемое значение: Tensor, матрица объектов n x D.
        """
        # ваш код здесь
        return self.decoder(z)

    def batch_loss(self, batch):
        """
        Вычисляет функцию потерь по батчу - усреднение функции потерь
        по объектам батча.
        Функция потерь по объекту- сумма L2-ошибки восстановления по батчу и
        L2 регуляризации скрытых представлений с весом 1.
        Возвращаемое значение должно быть дифференцируемо по параметрам модели (!).
        Вход: batch, Tensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - функция потерь по батчу.
        """
        # ваш код здесь
        z = self.encoder(batch)
        x = self.decoder(z)
        n = batch.shape[0]
        return torch.sum((x-batch)**2)/n + torch.sum(z**2)/n

    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы объектов x. Использует стандартное нормальное
        распределение в пространстве представлений.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        # ваш код здесь
        return torch.bernoulli(self.decoder(torch.randn([num_samples, self.d], requires_grad=True)))

def log_mean_exp(data):
    """
    Возвращает логарифм среднего по последнему измерению от экспоненты данной матрицы.
    Подсказка: не забывайте про вычислительную стабильность!
    Вход: mtx, Tensor - тензор размера n_1 x n_2 x ... x n_K.
    Возвращаемое значение: Tensor, тензор размера n_1 x n_2 x ,,, x n_{K - 1}.
    """
    # ваш код здесь
    return torch.logsumexp(data, dim = -1) - torch.log(torch.Tensor([data.shape[-1]]))


def log_likelihood(x_true, x_distr):
    """
    Вычисляет логарфм правдоподобия объектов x_true для индуцированного
    моделью покомпонентного распределения Бернулли.
    Каждому объекту из x_true соответствуют K сэмплированных распределений
    на x из x_distr.
    Требуется вычислить оценку логарифма правдоподобия для каждого объекта.
    Подсказка: не забывайте про вычислительную стабильность!
    Подсказка: делить логарифм правдоподобия на число компонент объекта не надо.

    Вход: x_true, Tensor - матрица объектов размера n x D.
    Вход: x_distr, Tensor - тензор параметров распределений Бернулли
    размера n x K x D.
    Выход: Tensor, матрица размера n x K - оценки логарифма правдоподобия
    каждого сэмпла.
    """
    # ваш код здесь
    n = x_distr.shape[0]
    K = x_distr.shape[1]
    D = x_distr.shape[2]
    
    T1 = torch.log(x_distr)*x_true.view([n, 1, D])
    
    T2 = torch.log(1-x_distr)*(1 - x_true).view([n, 1, D])
    
    return torch.sum(T1 + T2, dim = 2)




def kl(q_distr, p_distr):
    """
    Вычисляется KL-дивергенция KL(q || p) между n парами гауссиан.
    Вход: q_distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: p_distr, tuple(Tensor, Tensor). Аналогично.
    Возвращаемое значение: Tensor, вектор размерности n, каждое значение которого - 
    - KL-дивергенция между соответствующей парой распределений.
    """
    p_mu, p_sigma = p_distr
    q_mu, q_sigma = q_distr
    # ваш код здесь
    D_KL = torch.sum((q_sigma/p_sigma)**2, dim = 1)
    D_KL -= p_mu.shape[1]
    D_KL += 2*torch.sum(torch.log(p_sigma), dim = 1) - 2*torch.sum(torch.log(q_sigma), dim = 1)
    D_KL += torch.sum((p_mu-q_mu)*(p_mu-q_mu)/(p_sigma**2), dim = 1)
    return 0.5*D_KL


class VAE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.proposal_network = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
        )
        self.proposal_mu_head = nn.Linear(200, self.d)
        self.proposal_sigma_head = nn.Linear(200, self.d)
        self.generative_network = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def proposal_distr(self, x):
        """
        Генерирует предложное распределение на z.
        Подсказка: областью значений sigma должны быть положительные числа.
        Для этого при генерации sigma следует использовать softplus (!) в качестве
        последнего преобразования.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        # ваш код здесь
        proposal = self.proposal_network(x)
        mu = self.proposal_mu_head(proposal)
        sigma = torch.nn.Softplus()(self.proposal_sigma_head(proposal))
        return mu, sigma

    def prior_distr(self, n):
        """
        Генерирует априорное распределение на z.
        Вход: n, int - число распределений.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        # ваш код здесь
        mu = torch.zeros([n, self.d])
        sigma = torch.ones([n, self.d])
        return mu, sigma

    def sample_latent(self, distr, K=1):
        """
        Генерирует сэмплы из гауссовского распределения на z.
        Сэмплы должны быть дифференцируемы по параметрам распределения!
        Вход: distr, tuple(Tensor, Tensor). Каждое Tensor - матрица размера n x d.
        Первое - mu, второе - sigma.
        Вход: K, int - число сэмплов для каждого объекта.
        Возвращаемое значение: Tensor, матрица размера n x K x d.
        """
        # ваш код здесь
        MU, SIGMA = distr
        return torch.randn([MU.shape[0], K, MU.shape[1]], requires_grad=True)*SIGMA.view([SIGMA.shape[0], 1, SIGMA.shape[1]]) + MU.view([MU.shape[0], 1, MU.shape[1]])


    def generative_distr(self, z):
        """
        По матрице латентных представлений z возвращает матрицу параметров
        распределения Бернулли для сэмплирования объектов x.
        Вход: z, Tensor - тензор n x K x d латентных представлений.
        Возвращаемое значение: Tensor, тензор параметров распределения
        Бернулли размера n x K x D.
        """
        # ваш код здесь
        epsilon = 0.01
        out = self.generative_network(z)
        
        mask0 = (out < epsilon).float()
        MASK0 = (out > epsilon).float()
        mask1 = (out > 1-epsilon).float()
        MASK1 = (out < 1-epsilon).float()
        
        return out*MASK0*MASK1 + mask0*epsilon + mask1*(1-epsilon)


    def batch_loss(self, batch):
        """
        Вычисляет вариационную нижнюю оценку логарифма правдоподобия по батчу.
        Вариационная нижняя оценка должна быть дифференцируема по параметрам модели (!),
        т. е. надо использовать репараметризацию.
        Требуется вернуть усреднение вариационных нижних оценок объектов батча.
        Вход: batch, FloatTensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - вариационная нижняя оценка логарифма
        правдоподобия по батчу.
        """
        # ваш код здесь
        propos_distr = self.proposal_distr(batch)
        pri_distr = self.prior_distr(batch.shape[0])
        
        x_true = batch
        x_distr = self.generative_distr(self.sample_latent(propos_distr))
        
        return torch.mean(torch.mean(log_likelihood(x_true, x_distr), dim = 1) - kl(propos_distr, pri_distr), dim = 0)


    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы из индуцируемого моделью распределения на объекты x.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        # ваш код здесь
        return torch.bernoulli(self.generative_distr(self.sample_latent(self.prior_distr(1), K=num_samples)).view([num_samples, -1]))



def gaussian_log_pdf(distr, samples):
    """
    Функция вычисляет логарифм плотности вероятности в точке относительно соответствующего
    нормального распределения, заданного покомпонентно своими средним и среднеквадратичным отклонением.
    Вход: distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: samples, Tensor - тензор размера n x K x d сэмплов в скрытом пространстве.
    Возвращаемое значение: Tensor, матрица размера n x K, каждый элемент которой - логарифм
    плотности вероятности точки относительно соответствующего распределения.
    """
    mu, sigma = distr
    # ваш код здесь
    f1 = torch.sum(((samples - mu.view([mu.shape[0], 1, mu.shape[1]]))**2)/sigma.view([sigma.shape[0], 1, sigma.shape[1]])**2, dim = 2)
    f2 = mu.shape[1]*(math.log(2) + math.log(math.pi))
    f3 = torch.sum(torch.log(sigma), dim = 1).view(sigma.shape[0], 1)
    return -0.5*(f1 + f2) - f3




def compute_log_likelihood_monte_carlo(batch, model, K):
    """
    Функция, оценку логарифма правдоподобия вероятностной модели по батчу методом Монте-Карло.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памяти.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    # ваш код здесь
    pri_distr = model.prior_distr(batch.shape[0])
    
    x_true = batch
    x_distr = model.generative_distr(model.sample_latent(pri_distr, K = K))
    return float(torch.mean(log_mean_exp(log_likelihood(x_true, x_distr)), dim = 0))


def compute_log_likelihood_iwae(batch, model, K):
    """
    Функция, оценку IWAE логарифма правдоподобия вероятностной модели по батчу.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памятыи.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, proposal_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    # ваш код здесь
    pri_distr = model.prior_distr(batch.shape[0])
    propos_distr = model.proposal_distr(batch)
    
    x_true = batch
    z_latent = model.sample_latent(propos_distr, K = K)
    x_distr = model.generative_distr(z_latent)
    
    LogLikelihood = log_likelihood(x_true, x_distr)
    GussLogPDF_prior = gaussian_log_pdf(pri_distr, z_latent)
    GussLogPDF_propos = gaussian_log_pdf(propos_distr, z_latent)
    
    return float(torch.mean(log_mean_exp(LogLikelihood + GussLogPDF_prior - GussLogPDF_propos), dim = 0))

    

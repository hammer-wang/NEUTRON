import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical, MixtureSameFamily, Independent
from torch.nn.modules.activation import Softplus

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim):
        super().__init__()

        self.n_components = n_components

        self.base = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Tanh()
        )

        self.std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Softplus()
        )

        self.weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_components),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.base(x)
        means = self.mean(h)
        stds = self.std(h)
        stds = torch.clamp(stds, min=1e-5)
        weights = self.weight(h)

        return means, stds, weights

    def loss(self, out, y):
        means, stds, weights = out
        batch_size = len(means)

        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(weights)
        gmm = MixtureSameFamily(mix, comp)
        likelihood = gmm.log_prob(y)

        return -likelihood.mean()

    def sample(self, x):
        means, stds, weights = self.forward(x)
         # prevent invalid values due to computational instability
        batch_size = len(means)
        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(weights)
        gmm = MixtureSameFamily(mix, comp)
        samples = torch.clamp(gmm.sample(), min=-1, max=1)
        return samples

class MultitaskMixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim, alpha=1):
        super().__init__()

        self.n_components = n_components
        self.alpha = alpha

        self.base = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh ()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mean = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Tanh()
        )

        self.std = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Softplus()
        )

        self.weight = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_components)
        )

        self.classification_criterion = nn.BCEWithLogitsLoss()

    def forward(self, x, z, z_, e):
        h = self.base(x)

        # for accurate color, use the target, otherwise use the true color
        # z_mix = z * e.view(-1, 1) + (1 - e.view(-1, 1)) * z_
        z_mix = z_ # true color
        means = self.mean(torch.cat((h, z_mix), dim=1))
        stds = self.std(torch.cat((h, z_mix), dim=1))
        weights = self.weight(torch.cat((h, z_mix), dim=1))

        stds = torch.clamp(stds, min=1e-5)
        # weights = torch.clamp(weights, min=1e-5, max=1)

        # classifier always takes the target color as its input
        logits = self.classifier(torch.cat((h, z), dim=1))

        return means, stds, weights, logits

    def loss(self, out, y, e):
        '''
        Args:
            out: output from the multitask network
            y: thickness target
            e: binary label for whether DeltaE_2000 <= 2.0
        '''
        means, stds, weights, logits = out
        batch_size = len(means)

        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(logits = weights)
        gmm = MixtureSameFamily(mix, comp)
        likelihood = gmm.log_prob(y)
        
        labeled_indices = e != -1

        return -likelihood.mean() + self.alpha * self.classification_criterion(torch.squeeze(logits[labeled_indices]), e[labeled_indices])

    def sample(self, x, z):
        # means, stds, weights, logits = self.forward(x)
        batch_size = len(x)
        h = self.base(x)
        
        means = self.mean(torch.cat((h, z), dim=1))
        stds = self.std(torch.cat((h, z), dim=1))
        weights = self.weight(torch.cat((h, z), dim=1))
        stds = torch.clamp(stds, min=1e-5)
        logits = self.classifier(torch.cat((h, z), dim=1))
        
        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(logits = weights)
        gmm = MixtureSameFamily(mix, comp)
        samples = gmm.sample()
        
        return samples, torch.sigmoid(logits) >= 0.5

    def mean_pred(self, x, z):
        # means, stds, weights, logits = self.forward(x)
        batch_size = len(x)
        h = self.base(x)
        
        means = self.mean(torch.cat((h, z), dim=1))
        stds = self.std(torch.cat((h, z), dim=1))
        weights = self.weight(torch.cat((h, z), dim=1))
        stds = torch.clamp(stds, min=1e-6, max=1e-5)
        
        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(logits = weights)
        gmm = MixtureSameFamily(mix, comp)
        samples = gmm.sample()
        
        return samples


class MultitaskMixtureDensityNetworkShared(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim, alpha=1):
        super().__init__()

        self.dim_out = dim_out
        self.n_components = n_components
        self.alpha = alpha

        self.base = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh ()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mixture_network = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.means = nn.Sequential(
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Tanh()
        )

        self.stds = nn.Sequential(
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Softplus()
        )

        self.weights = nn.Sequential(
            nn.Linear(hidden_dim, n_components),
            nn.Softmax(dim=1)
        )



        # self.std = nn.Sequential(
        #     nn.Linear(hidden_dim + 3, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, dim_out * n_components),
        #     nn.Softplus()
        # )

        # self.weight = nn.Sequential(
        #     nn.Linear(hidden_dim + 3, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, n_components),
        #     nn.Softmax(dim=1)
        # )

        self.classification_criterion = nn.BCEWithLogitsLoss()

    def forward(self, x, z, z_, e):
        '''
        Args:
            x: input data
            z: target color
            z_: designed color
            e: labels
        '''
        h = self.base(x)

        # for accurate color, use the target, otherwise use the true color
        # z_mix = z * e.view(-1, 1) + (1 - e.view(-1, 1)) * z_
        # z_mix = z_ # true color
        # means = self.mean(torch.cat((h, z_mix), dim=1))
        # stds = self.std(torch.cat((h, z_mix), dim=1))
        # weights = self.weight(torch.cat((h, z_mix), dim=1))

        mdn_out = self.mixture_network(torch.cat((h, z_), dim=1))

        means = self.means(mdn_out)
        stds = self.stds(mdn_out)
        weights = self.weights(mdn_out)

        # stds = torch.clamp(stds, min=1e-5)

        # classifier always takes the target color as its input
        logits = self.classifier(torch.cat((h, z), dim=1))

        return means, stds, weights, logits

    def loss(self, out, y, e):
        '''
        Args:
            out: output from the multitask network
            y: thickness target
            e: binary label for whether DeltaE_2000 <= 2.0
        '''
        means, stds, weights, logits = out
        batch_size = len(means)
        
        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)

        weights = torch.clamp(weights, min=1e-6)
        stds = torch.clamp(stds, min=1e-6)

        mix = Categorical(weights)
        gmm = MixtureSameFamily(mix, comp)
        likelihood = gmm.log_prob(y)
        
        labeled_indices = e != -1

        return -likelihood.mean() + self.alpha * self.classification_criterion(torch.squeeze(logits[labeled_indices]), e[labeled_indices])

    def sample(self, x, z):
        # means, stds, weights, logits = self.forward(x)
        batch_size = len(x)

        h = self.base(x)

        mdn_out = self.mixture_network(torch.cat((h, z), dim=1))

        means = self.means(mdn_out)
        stds = self.stds(mdn_out)
        weights = self.weights(mdn_out)

        weights = torch.clamp(weights, min=1e-6)
        stds = torch.clamp(stds, min=1e-6)

        # h = self.base(x)
        
        # means = self.mean(torch.cat((h, z), dim=1))
        # stds = self.std(torch.cat((h, z), dim=1))
        # weights = self.weight(torch.cat((h, z), dim=1))
        # stds = torch.clamp(stds, min=1e-5)
        logits = self.classifier(torch.cat((h, z), dim=1))
        
        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(weights)
        gmm = MixtureSameFamily(mix, comp)
        samples = gmm.sample()
        
        return samples, torch.sigmoid(logits) >= 0.5

    def mean_pred(self, x, z):
        # means, stds, weights, logits = self.forward(x)
        batch_size = len(x)

        h = self.base(x)
        mdn_out = self.mixture_network(torch.cat((h, z), dim=1))
        means = torch.tanh(mdn_out[:, :self.dim_out * self.n_components])
        stds = nn.functional.softplus(mdn_out[:, self.dim_out * self.n_components:2 * self.dim_out * self.n_components])
        weights = nn.functional.softmax(mdn_out[:, 2 * self.dim_out * self.n_components:], dim=1)
        stds = torch.clamp(stds, min=1e-6, max=1e-5)
        
        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(weights)
        gmm = MixtureSameFamily(mix, comp)
        samples = gmm.sample()
        
        return samples

class MixtureDensityNetworkCNN(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim):
        super().__init__()

        self.n_components = n_components

        self.base = nn.Sequential(
            nn.Conv1d(10, 64, 3, stride=1),
            nn.ELU(),
            nn.Conv1d(64, 64, 3, stride=1),
            nn.ELU(),
            nn.Conv1d(64, 64, 3, stride=1),
            nn.ELU(),
            nn.Conv1d(64, 32, 3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.Tanh(),
        )

        self.mean = nn.Sequential(
            nn.Linear(355, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Tanh(),
        )

        self.std = nn.Sequential(
            nn.Linear(355, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out * n_components),
            nn.Softplus()
        )

        self.weight = nn.Sequential(
            nn.Linear(355, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_components),
            nn.Softmax(dim=1)
        )

    def forward(self, x, y_):
        h = self.base(x).view(len(x), -1)
        h = torch.cat((h, y_), dim=1)
        means = self.mean(h)
        stds = self.std(h)
        stds = torch.clamp(stds, min=1e-5)
        weights = self.weight(h)

        return means, stds, weights

    def loss(self, out, y):
        means, stds, weights = out
        batch_size = len(means)

        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(weights)
        gmm = MixtureSameFamily(mix, comp)
        likelihood = gmm.log_prob(y)

        return -likelihood.mean()

    def sample(self, x, y_):
        '''
        x: nk data
        y_: target color
        '''
        means, stds, weights = self.forward(x, y_)
         # prevent invalid values due to computational instability
        batch_size = len(means)
        comp = Independent(Normal(means.view(batch_size, self.n_components, -1), stds.view(batch_size, self.n_components, -1)), 1)
        mix = Categorical(weights)
        gmm = MixtureSameFamily(mix, comp)
        samples = torch.clamp(gmm.sample(), min=-1, max=1)
        return samples

# class MixtureDensityNetwork(nn.Module):
#     """
#     Mixture density network.
#     [ Bishop, 1994 ]
#     Parameters
#     ----------
#     dim_in: int; dimensionality of the covariates
#     dim_out: int; dimensionality of the response variable
#     n_components: int; number of components in the mixture model
#     """
#     def __init__(self, dim_in, dim_out, n_components, hidden_dim):
#         super().__init__()
#         self.pi_network = CategoricalNetwork(dim_in, n_components, hidden_dim)
#         self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
#                                                        n_components, hidden_dim)

#     def forward(self, x):
#         return self.pi_network(x), self.normal_network(x)

#     def loss(self, out, y):
#         pi, normal = out
#         loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
#         loglik = torch.sum(loglik, dim=2)
#         loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
#         return loss.mean()

#     def sample(self, x):
#         pi, normal = self.forward(x)
#         samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
#         return samples


# class MixtureDiagNormalNetwork(nn.Module):

#     def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
#         super().__init__()
#         self.n_components = n_components
#         if hidden_dim is None:
#             hidden_dim = in_dim
#         self.network = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, 2 * out_dim * n_components)
#         )

#     def forward(self, x):
#         params = self.network(x)
#         mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
#         mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
#         sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
#         return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

# class CategoricalNetwork(nn.Module):

#     def __init__(self, in_dim, out_dim, hidden_dim=None):
#         super().__init__()
#         if hidden_dim is None:
#             hidden_dim = in_dim
#         self.network = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, out_dim)
#         )

#     def forward(self, x):
#         params = self.network(x)
#         return OneHotCategorical(logits=params)

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import time
import math
from utils import *
from monotonic_gru import MonotonicGru, MonotonicGruCell
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_binary_gumbel(shape, eps=1e-12):
    U = torch.rand(shape).to(device)
    return torch.log(U + eps) - torch.log(1-U + eps)


def gumbel_binary_softmax_sample(p, temperature, eps=1e-12):
    logits = torch.log(p+eps) - torch.log(1-p + eps)
    y = logits + sample_binary_gumbel(logits.size())
    m = nn.Sigmoid()
    return m(y / temperature)


def st(y):
    """
    ST-gumple-softmax
    input: [*, 1]
    return: flatten --> [*, 1] an one-hot vector
    """
    shape = y.size()
    flag = y > 0.5
    y_hard = torch.zeros_like(y).to(device)
    y_hard[flag] = 1.
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class MNSS(nn.Module):
    def __init__(self, mapper, embedding_dimensions, data_dir=None, use_loc_ind=False, shared_layer_dim=512, hidden_dim=512, dropout=0.25, num_gru_layers=1, zero_prior=True, pre_trained=False, gumbel=False):
        super().__init__()
        # model can be 'gru', 'm_gru'
        self.use_loc_ind = use_loc_ind
        self.hidden_dim = hidden_dim
        self.embedding_dimensions = embedding_dimensions
        self.company_embed = nn.Embedding(
            max(mapper['companies'].values())+1, embedding_dimensions['companies'])
        self.school_embed = nn.Embedding(
            max(mapper['schools'].values())+1, embedding_dimensions['schools'])
        self.degree_embed = nn.Embedding(
            max(mapper['degrees'].values())+1, embedding_dimensions['degrees'])
        self.time_embed = nn.Linear(1, embedding_dimensions['times'])
        self.interval_embed = nn.Linear(1, embedding_dimensions['intervals'])
        if use_loc_ind:
            self.locality_embed = nn.Embedding(
                max(mapper['localities'].values())+1, embedding_dimensions['locality'])
            self.industry_embed = nn.Embedding(
                max(mapper['industries'].values())+1, embedding_dimensions['industry'])

        self.majors_embed = nn.Embedding(
            max(mapper['majors'].values())+1, embedding_dimensions['majors'])
        self.titles_embed = nn.Embedding(
            max(mapper['occupations'].values())+1, embedding_dimensions['occupations'])

        temp_dim = embedding_dimensions['industry'] + \
            embedding_dimensions['locality']
        self.total_input_dimension = sum(
            embedding_dimensions.values()) - temp_dim-embedding_dimensions['skills']
        self.total_fixed_dimension = temp_dim if use_loc_ind else 0
        self.num_gru_layers = num_gru_layers

        # Model parameterl
        self.num_skill_label = max(mapper['skills'].values())+1
        self.posterior_rnn = MonotonicGru(input_size=self.total_input_dimension,
                                          hidden_size=hidden_dim, num_layers=num_gru_layers, batch_first=False)
        self.prior_rnn = MonotonicGru(input_size=self.total_input_dimension,
                                      hidden_size=hidden_dim, num_layers=num_gru_layers, batch_first=False)
        self.skill_cell = MonotonicGruCell(
            input_size=self.total_input_dimension, hidden_size=hidden_dim)
        self.skill_embed_cell = nn.Linear(
            self.num_skill_label, self.total_input_dimension)

        # Final Parameters
        self.intermediate_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim+self.total_fixed_dimension, shared_layer_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.company_final_layers = nn.Sequential(
            nn.Linear(shared_layer_dim, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, embedding_dimensions['companies'])
        )
        self.title_final_layers = nn.Sequential(
            nn.Linear(shared_layer_dim, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, embedding_dimensions['occupations'])
        )
        self.skill_final_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim+self.total_fixed_dimension,
                      embedding_dimensions['skills'])
        )
        self.skill_embed = nn.Linear(
            embedding_dimensions['skills'], max(mapper['skills'].values())+1)
        self.zero_prior = zero_prior
        if self.zero_prior is False:
            self.logit_prior = nn.Parameter(torch.zeros(1, hidden_dim) - 4)
        if pre_trained:
            self.company_embed.weight.data = torch.load(
                data_dir+'company_pretrained.pt')
            self.titles_embed.weight.data = torch.load(
                data_dir+'title_pretrained.pt')
            self.skill_embed.weight.data = torch.load(
                data_dir+'skill_pretrained.pt')
            self.skill_embed.bias.data = torch.zeros(
                max(mapper['skills'].values())+1)

        self.gumbel = gumbel

    def forward(self, batch, negative_batch, alpha=1., beta=1.):
        # unpack data
        L, T, B, job_mask, edu_mask, locality, industry, times, intervals, schools, degrees, majors,  companies, titles, skill_label = tuple(
            i.to(device) if isinstance(i, torch.Tensor) else i for i in batch)
        # setup mask
        company_mask = companies > 0  # T, B
        title_mask = titles > 0  # T, B
        total_time_steps = job_mask.sum()

        if self.use_loc_ind is False:
            del locality, industry
        # unpack negative data
        negative_companies, negative_titles, company_qt, title_qt = tuple(
            i.to(device) if isinstance(i, torch.Tensor) else i for i in negative_batch)

        # Majors
        majors_embed = self.majors_embed(majors)  # T, B, 20
        # Occupation
        titles_embed = self.titles_embed(titles)  # T,B, 20
        negative_titles_embed = self.titles_embed(
            negative_titles)  # T, B, N, 20

        all_titles_embed = torch.cat([titles_embed.unsqueeze(
            2), negative_titles_embed], dim=2)  # T, B, N+1, 20

        # companies
        companies_embed = self.company_embed(companies)  # T, B, 20
        negative_companies_embed = self.company_embed(
            negative_companies)  # T, B, N , 20
        all_companies_embed = torch.cat([companies_embed.unsqueeze(
            2), negative_companies_embed], dim=2)  # T, B, N+1, 20

        # Embed
        if self.use_loc_ind:
            # locality
            locality_embed = self.locality_embed(locality).expand(T, B, -1)
            industry_embed = self.industry_embed(industry).expand(T, B, -1)
            #locality_embed, industry_embed
            input_fixed_embeddings = torch.cat(
                [locality_embed, industry_embed], dim=-1)  # T, B, H1
        # schools
        schools_embed = self.school_embed(schools)
        degrees_embed = self.degree_embed(degrees)
        times = (times.float().unsqueeze(2)-180.)/30.
        times_embed = self.time_embed(times)
        intervals_ = (intervals.float().unsqueeze(2) - 5.)/5.
        intervals_embed = self.interval_embed(intervals_)

        # input Embedding
        input_edu_embeddings = torch.cat(
            [degrees_embed, schools_embed, majors_embed], dim=-1)*edu_mask.unsqueeze(2)  # T, B, H1
        input_job_embeddings = torch.cat(
            [companies_embed, titles_embed], dim=-1)*job_mask.unsqueeze(2)  # T, B, H1
        input_embeddings = torch.cat(
            [input_edu_embeddings, input_job_embeddings, intervals_embed, times_embed], dim=-1)

        # Run RNN #T,B, H and B,H
        if self.zero_prior is False:
            m = nn.Sigmoid()
            prior = m(self.logit_prior).expand(B, -1)
        else:
            prior = torch.zeros(B, self.hidden_dim).to(device)

        prior_hidden_states, prior_last_states = self.run_rnn(
            input_embeddings, self.prior_rnn, L, B, T,  hidden=prior)
        # append init states to prior
        # torch.zeros(1,B, prior_hidden_states.shape[-1]).to(device)
        init_state = prior.unsqueeze(0)
        prior_hidden_states = torch.cat(
            [init_state, prior_hidden_states[:-1, :, :]], dim=0)
        if self.gumbel == 1:
            prior_hidden_states = st(prior_hidden_states)
            prior_last_states = st(prior_last_states)

        # MLE Loss
        if self.use_loc_ind:
            prior_augmented_hidden_states = torch.cat(
                [prior_hidden_states, input_fixed_embeddings], dim=-1)
            augmented_prior_last_states = torch.cat(
                [prior_last_states, input_fixed_embeddings[0, :, :]], dim=-1)
        else:
            prior_augmented_hidden_states = prior_hidden_states
            augmented_prior_last_states = prior_last_states
        #####
        shared_output = self.intermediate_layers(
            prior_augmented_hidden_states)  # T, B, 100
        # predict skills (B, 100)
        skill_shared_layer = self.skill_embed(self.skill_final_layers(
            augmented_prior_last_states))  # B, number_skills
        # loss
        skill_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        skill_loss = (skill_loss_fn(skill_shared_layer, skill_label)).mean(
            dim=-1).sum()  # B, S

        # predict companies (T, B, 100)
        company_shared_layer = self.company_final_layers(
            shared_output)  # T, B , 100
        # loss
        company_loss = (restricted_sigmoid(company_shared_layer,
                                           all_companies_embed, company_qt, company_mask)).sum()
        # predict title (T, B, 100)
        title_shared_layer = self.title_final_layers(
            shared_output)  # T, B , 100
        # loss
        title_loss = (restricted_sigmoid(title_shared_layer,
                                         all_titles_embed, title_qt, title_mask)).sum()

        loss = (company_loss+title_loss)/total_time_steps+skill_loss/B

        # Add fixed info
        if self.training and alpha > 0.:
            posterior_hidden_states, posterior_last_states = self.run_rnn(
                input_embeddings, self.posterior_rnn, L, B, T, self.model)
            if self.zero_prior:
                posterior_hidden_states = torch.cat(
                    [init_state, posterior_hidden_states[1:, :, :]], dim=0)
            # Skill cell
            posterior_last_states = self.skill_cell(
                self.skill_embed_cell(skill_label), posterior_last_states)
            if self.gumbel == 1:
                posterior_hidden_states = st(posterior_hidden_states)
                posterior_last_states = st(posterior_last_states)
            if self.use_loc_ind:
                posterior_augmented_hidden_states = torch.cat(
                    [posterior_hidden_states, input_fixed_embeddings], dim=-1)
                augmented_posterior_last_states = torch.cat(
                    [posterior_last_states, input_fixed_embeddings[0, :, :]], dim=-1)
            else:
                posterior_augmented_hidden_states = posterior_hidden_states
                augmented_posterior_last_states = posterior_last_states
            #####
            shared_output_1 = self.intermediate_layers(
                posterior_augmented_hidden_states)  # T, B, 100
            # predict skills (B, 100)
            skill_shared_layer_1 = self.skill_embed(self.skill_final_layers(
                augmented_posterior_last_states))  # B, number_skills
            # loss
            skill_loss_1 = (skill_loss_fn(skill_shared_layer_1,
                                          skill_label)).mean(dim=-1).sum()  # B, S

            # predict companies (T, B, 100)
            company_shared_layer_1 = self.company_final_layers(
                shared_output_1)  # T, B , 100
            # loss
            company_loss_1 = (restricted_sigmoid(
                company_shared_layer_1, all_companies_embed, company_qt, company_mask)).sum()
            # predict title (T, B, 100)
            title_shared_layer_1 = self.title_final_layers(
                shared_output_1)  # T, B , 100
            # loss
            title_loss_1 = (restricted_sigmoid(
                title_shared_layer_1, all_titles_embed, title_qt, title_mask)).sum()

            reconstruction_loss = (
                company_loss_1+title_loss_1)/total_time_steps+skill_loss_1/B
            kl_loss_1 = kl_(posterior_hidden_states, prior_hidden_states, L)
            kl_loss_2 = kl_(posterior_last_states, prior_last_states)
            kl_loss = kl_loss_1.sum()/total_time_steps + kl_loss_2.sum()/B
            loss = alpha * (reconstruction_loss + beta * kl_loss) + loss

        return loss

    def run_rnn(self, data_embed, rnn, data_length, batch_size, max_len,  hidden=None):
        output, ht = rnn(data_embed, data_length, hidden=hidden)
        return output, ht

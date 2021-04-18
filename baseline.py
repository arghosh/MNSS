import torch
import numpy as np
import torch.nn as nn
from dataset import data_generator
from torch.optim import Adam
import time
import math
from utils import *
from monotonic_gru import MonotonicGru

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Baseline(nn.Module):
    def __init__(self, mapper, embedding_dimensions, data_dir=None, use_loc_ind=False, shared_layer_dim=512, hidden_dim=512, dropout=0.25, model='nemo', num_gru_layers=1, pre_trained=False):
        super().__init__()
        # model can be 'gru', 'm_gru'
        self.use_loc_ind = use_loc_ind
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

        # Model parameter
        self.model = model
        if self.model == 'nemo':
            self.rnn = nn.GRU(input_size=self.total_input_dimension, hidden_size=hidden_dim,
                              num_layers=num_gru_layers, batch_first=False, bidirectional=False)
        elif self.model in {'nss'}:
            self.rnn = MonotonicGru(input_size=self.total_input_dimension,
                                    hidden_size=hidden_dim, num_layers=num_gru_layers, batch_first=False)

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
        if self.model == 'nemo':
            self.skill_final_layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim+self.total_fixed_dimension,
                          embedding_dimensions['skills'])
            )
            self.skill_embed = nn.Linear(
                embedding_dimensions['skills'], max(mapper['skills'].values())+1)
        elif self.model == 'nss':
            self.skill_final_layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim+self.total_fixed_dimension,
                          embedding_dimensions['skills'])
            )
            self.skill_embed = nn.Linear(
                embedding_dimensions['skills'], max(mapper['skills'].values())+1)

        self.num_skill_label = max(mapper['skills'].values())+1
        if pre_trained:
            self.company_embed.weight.data = torch.load(
                data_dir+'company_pretrained.pt')
            self.titles_embed.weight.data = torch.load(
                data_dir+'title_pretrained.pt')
            self.skill_embed.weight.data = torch.load(
                data_dir+'skill_pretrained.pt')
            self.skill_embed.bias.data = torch.zeros(
                max(mapper['skills'].values())+1)

    def forward(self, batch, negative_batch):
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
        hidden_states, last_states = self.run_rnn(
            input_embeddings, self.rnn, L, B, T, self.model)

        # Add fixed info
        if self.use_loc_ind:
            augmented_hidden_states = torch.cat(
                [hidden_states, input_fixed_embeddings], dim=-1)
            augmented_last_states = torch.cat(
                [last_states, input_fixed_embeddings[0, :, :]], dim=-1)
        else:
            augmented_hidden_states = hidden_states
            augmented_last_states = last_states

        shared_output = self.intermediate_layers(
            augmented_hidden_states)  # T, B, 100

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

        # predict skills (B, 100)
        skill_shared_layer = self.skill_embed(
            self.skill_final_layers(augmented_last_states))  # B, number_skills
        # loss
        skill_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        skill_loss = (skill_loss_fn(skill_shared_layer, skill_label)).mean(
            dim=-1).sum()  # B, S

        total_loss = (company_loss+title_loss)/total_time_steps+skill_loss/B

        return total_loss

    def run_rnn(self, data_embed, rnn, data_length, batch_size, max_len, model):
        if model == 'nemo':
            packed_data = torch.nn.utils.rnn.pack_padded_sequence(
                data_embed, lengths=(data_length), batch_first=False, enforce_sorted=False)
            packed_output, ht = rnn(packed_data)
            ht = ht.permute(1, 2, 0).contiguous().reshape(batch_size, -1)
            output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=False)
        elif model in {'nss'}:
            output, ht = rnn(data_embed, data_length)
        init_state = torch.zeros(1, batch_size, output.shape[-1]).to(device)
        # append init states
        output = torch.cat([init_state, output[:-1, :, :]], dim=0)
        return output, ht

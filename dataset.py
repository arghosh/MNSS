import numpy as np
import torch
from torch.utils import data
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class custom_collate(object):
    def __init__(self, negative_mapper, negative_count, max_skills):
        self.negative_count = negative_count
        self.negative_mapper = negative_mapper
        self.max_skills = max_skills
        self.sample = self.negative_count > 0 and self.negative_mapper is not None

    def __call__(self, batch):
        max_skills = self.max_skills
        lengths = [len(data['degrees']) for data in batch]
        max_length = max(lengths)
        batch_size = len(lengths)

        # T,N
        job_mask = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            job_mask[:lengths[i], i] = batch[i]['job_mask']
        edu_mask = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            edu_mask[:lengths[i], i] = batch[i]['edu_mask']

        # 1,N
        locality = torch.cat([data['locality'].reshape(1, -1)
                              for data in batch], dim=1)
        industry = torch.cat([data['industry'].reshape(1, -1)
                              for data in batch], dim=1)

        # T,N
        times = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            times[:lengths[i], i] = batch[i]['times']
        intervals = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            intervals[:lengths[i], i] = batch[i]['intervals']

        # T,N
        companies = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            companies[:lengths[i], i] = batch[i]['companies']

        # negative companies T, N, C
        if self.sample:
            negative_companies_idx = torch.multinomial(
                self.negative_mapper['companies']['weights'], batch_size * max_length * self.negative_count, replacement=True).long()
            negative_companies = self.negative_mapper['companies']['values'][negative_companies_idx].reshape(
                max_length, batch_size, self.negative_count)
            company_qt = torch.log(self.negative_mapper['companies']['weights'][negative_companies_idx].reshape(
                max_length, batch_size, self.negative_count))

        # T, N
        schools = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            schools[:lengths[i], i] = batch[i]['schools']
        degrees = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            degrees[:lengths[i], i] = batch[i]['degrees']

        # T, N
        majors = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            majors[:lengths[i], i] = batch[i]['majors']

        # T,N
        occupations = torch.zeros(max_length, batch_size).long()
        for i in range(batch_size):
            occupations[:lengths[i], i] = batch[i]['occupations']

        # negative occupation
        if self.sample:
            negative_occupation_idx = torch.multinomial(
                self.negative_mapper['occupations']['weights'], batch_size * max_length * self.negative_count, replacement=True).long()
            # NTC
            negative_occupation = self.negative_mapper['occupations']['values'][negative_occupation_idx].reshape(
                max_length, batch_size, self.negative_count)
            title_qt = torch.log(self.negative_mapper['occupations']['weights'][negative_occupation_idx].reshape(
                max_length, batch_size, self.negative_count))

        #  S, N
        num_skills = [len(data['skills']) for data in batch]
        if max(num_skills) == 0:
            return None, None
        skill_label = torch.zeros(batch_size, max_skills) + 1e-3

        for i in range(batch_size):
            skill_label[i, batch[i]['skills']] = 1.

        batch = (lengths, max_length, batch_size, job_mask, edu_mask, locality, industry,
                 times, intervals, schools, degrees, majors,  companies, occupations, skill_label)

        if self.sample:
            negative_batch = (negative_companies,
                              negative_occupation, company_qt, title_qt)
        else:
            negative_batch = None
        return (batch, negative_batch)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, profiles):
        'Initialization'
        self.profiles = profiles

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.profiles)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        profile = self.profiles[index]
        out = {}
        # mask
        out['job_mask'] = torch.LongTensor(profile['job_mask'])
        out['edu_mask'] = torch.LongTensor(profile['edu_mask'])
        out['skill_mask'] = torch.LongTensor([profile['skill_mask']])

        # static info
        out['locality'] = torch.LongTensor(profile['localities'])
        out['industry'] = torch.LongTensor(profile['industries'])
        out['times'] = torch.LongTensor(profile['time'])
        out['intervals'] = torch.LongTensor(profile['intervals'])
        # dynamic fixed info
        out['companies'] = torch.LongTensor(profile['companies'])
        out['schools'] = torch.LongTensor(profile['schools'])
        out['degrees'] = torch.LongTensor(profile['degrees'])

        # dynamic info
        out['majors'] = torch.LongTensor(profile['majors'])
        out['occupations'] = torch.LongTensor(profile['occupations'])
        out['skills'] = torch.LongTensor(profile['skills'])

        # Load data and get label
        return out


def data_generator(path, negative_count=0, negative_mapper=None, start_=0., end_=1., batch_size=32, num_workers=0, shuffle=True, drop_last=True, pin_memory=True, power=0.125):
    profiles = open_json(path)
    n = len(profiles)
    start_index = int(n*start_)
    end_index = min(n, int(n*end_))
    max_skills = len(negative_mapper['skills']['names']) + 1
    if start_index > 0 or end_index < n:
        profiles = profiles[start_index:end_index]
    if negative_mapper is not None and negative_count > 0:
        mapper = {}
        for name in ['companies', 'occupations', 'skills']:
            mapper[name] = {'names': negative_mapper[name]['names']}
            mapper[name]['values'] = torch.tensor(
                negative_mapper[name]['values']).long()
            wf = torch.tensor(negative_mapper[name]['weights'])**power
            mapper[name]['weights'] = wf/wf.sum()
    else:
        mapper = None
    dataset = Dataset(profiles)
    collate_fn = custom_collate(mapper, negative_count, max_skills)

    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'collate_fn': collate_fn,
              'pin_memory': pin_memory,
              'drop_last': drop_last,
              'num_workers': num_workers}
    return data.DataLoader(dataset, **params)

'''
 # Copyright
 # 2023/4/28
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import json
import os
import random
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset,DataLoader
from utils import _transform
from PIL import Image
import warnings
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")

class load_data(object):
    def __init__(self,args):
        self.L_VWSD_dir = args.L_VWSD_dir
        self.official_data_dir = args.official_data_dir

        self.load_lvwsd_data()
        self.load_official_testdata()
        self.sense_supplement()

    def load_lvwsd_data(self):
        # load L-VWSD.data
        lvwsd_path = os.path.join(self.L_VWSD_dir,'main.data', 'T-VWSD_translation.data.json')
        with open(lvwsd_path, 'r', encoding='utf-8') as f:
            self.lvwsd_df = json.load(f)
        self.lvwsd_df = pd.DataFrame(self.lvwsd_df)

        imgid2senid_path = os.path.join(self.L_VWSD_dir,'main.data', 'imgid2senid.json')
        with open(imgid2senid_path, 'r', encoding='utf-8') as f:
            self.imgid2senid = json.load(f)

    def load_official_testdata(self):
        # load test data
        testdata_root = os.path.join(self.official_data_dir, 'test.data.v1.1.gold')
        en_data = open(os.path.join(testdata_root, 'en.test.data.txt'), 'r', encoding='utf-8').readlines()
        fa_data = open(os.path.join(testdata_root, 'fa.test.data.txt'), 'r', encoding='utf-8').readlines()
        it_data = open(os.path.join(testdata_root, 'it.test.data.txt'), 'r', encoding='utf-8').readlines()

        en_gold = open(os.path.join(testdata_root, 'en.test.gold.v1.1.txt'), 'r', encoding='utf-8').readlines()
        fa_gold = open(os.path.join(testdata_root, 'fa.test.gold.v1.1.txt'), 'r', encoding='utf-8').readlines()
        it_gold = open(os.path.join(testdata_root, 'it.test.gold.v1.1.txt'), 'r', encoding='utf-8').readlines()

        # integrate data
        test_image_dir = os.path.join(self.official_data_dir, 'images')
        self.entest_df = self.integrate_data(en_data,en_gold,test_image_dir)
        self.fatest_df = self.integrate_data(fa_data,fa_gold,test_image_dir)
        self.ittest_df = self.integrate_data(it_data, it_gold, test_image_dir)

    def integrate_data(self, data, gold, image_dir):
        df = []
        for i in range(len(data)):
            c = data[i].strip().split('\t')
            temp = {}
            temp['amb'] = c[0]
            temp['phrase'] = c[1]
            candidate_imgs = c[2:]
            img_paths = [os.path.join(image_dir, cand) for cand in candidate_imgs]
            temp['candidate_imgs'] = candidate_imgs
            temp['img_paths'] = img_paths
            temp['gold_img'] = gold[i].strip()
            df.append(temp)
        df = pd.DataFrame(df)
        return df

    def sense_supplement(self):
        # load gloss from chatgpt
        sense_dir = os.path.join(self.official_data_dir,'gloss_chatgpt')
        en_sense = open(os.path.join(sense_dir, 'en.txt'), 'r', encoding='utf-8').readlines()
        fa_sense = open(os.path.join(sense_dir, 'fa.txt'), 'r', encoding='utf-8').readlines()
        it_sense = open(os.path.join(sense_dir, 'it.txt'), 'r', encoding='utf-8').readlines()

        self.en_sense = self.process_sense(en_sense,'en')
        self.fa_sense = self.process_sense(fa_sense, 'fa')
        self.it_sense = self.process_sense(it_sense, 'it')

    def process_sense(self,sense,language):
        df = []
        if language == 'en':
            for c in sense:
                temp = {}
                c = c.strip().split('\t')
                temp['amb'] = c[0]
                temp['phrase'] = c[1]
                temp['sense_en'] = c[2].replace('\"','').lower()
                df.append(temp)
        else:
            for c in sense:
                temp = {}
                c = c.strip().split('\t')
                temp['amb'] = c[0]
                temp['amb_translation'] = c[1]
                temp['phrase'] = c[2]
                temp['phrase_translation'] = c[3]
                temp['sense_en'] = c[4].replace('\"','').lower()
                ori_sense = 'sense_' + language
                temp[ori_sense] = c[5].replace('\"', '').lower()
                df.append(temp)
        df = pd.DataFrame(df)
        return df

def get_img_vec(image_path):
    image = Image.open(image_path).convert("RGB")
    image_vec = _transform(image)
    image_vec = image_vec.unsqueeze(0)
    return image_vec

class train_dataset(Dataset):
    def __init__(self, data,args, mode):
        super(train_dataset, self).__init__()
        self.lvwsd_df = data.lvwsd_df
        self.imgid2senid = data.imgid2senid
        self.image_dir = os.path.join(args.L_VWSD_dir,'images')
        self.image_keys = list(data.imgid2senid.keys())
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.lvwsd_df)

    def image_supplement(self,image_ids,n):
        total = len(self.image_keys)
        for i in range(n):
            while 1:
                rand = random.randint(0,total-1)
                key = int(self.image_keys[rand])
                if key not in image_ids:
                    image_ids.append(key)
                    break
        return image_ids


    def __getitem__(self, item):
        ambiguous_word = self.lvwsd_df.loc[item]['word']
        word_language = self.lvwsd_df.loc[item]['word_language']
        sense_id = self.lvwsd_df.loc[item]['sense_id']

        en_concept = self.lvwsd_df.loc[item]['en_concept']
        fa_concept = self.lvwsd_df.loc[item]['fa_concept']
        it_concept = self.lvwsd_df.loc[item]['it_concept']
        concepts = [en_concept,fa_concept,it_concept]

        en_gloss = self.lvwsd_df.loc[item]['en_gloss']
        fa_gloss = self.lvwsd_df.loc[item]['fa_gloss']
        it_gloss = self.lvwsd_df.loc[item]['it_gloss']
        glosses = [en_gloss,fa_gloss,it_gloss]

        image_ids = self.lvwsd_df.loc[item]['image_ids']
        image_labels = [1] * len(image_ids) + [0] * (5-len(image_ids))
        if len(image_ids) < 5:
            image_ids = self.image_supplement(image_ids,5-len(image_ids))
        image_paths = [os.path.join(self.image_dir,str(id)+'.jpg') for id in image_ids]
        image_vecs = torch.vstack([get_img_vec(path) for path in image_paths])

        return ambiguous_word,word_language,sense_id,concepts,glosses,image_ids,image_vecs,image_labels

def collate_train_fn(batch):
    ambiguous_word,word_language,sense_id,concepts,glosses,image_ids,image_vecs,image_labels = zip(*batch)
    ambiguous_word,word_language,sense_id,concepts,glosses= list(ambiguous_word),list(word_language),list(sense_id),list(concepts),list(glosses)
    image_ids,image_vecs,image_labels = list(image_ids),torch.vstack(image_vecs),list(image_labels)
    return ambiguous_word,word_language,sense_id,concepts,glosses,image_ids,image_vecs,image_labels

class test_dataset(Dataset):
    def __init__(self, data, args, mode,input_pattern):
        super(test_dataset, self).__init__()
        if mode == 'test_en':
            self.test_data = data.entest_df
            self.sense_sup = data.en_sense
            self.language = 'en'
        elif mode == 'test_fa':
            self.test_data = data.fatest_df
            self.sense_sup = data.fa_sense
            self.language = 'fa'
        elif mode == 'test_it':
            self.test_data = data.ittest_df
            self.sense_sup = data.it_sense
            self.language = 'it'
        self.args = args

        if input_pattern == 1:
            self.input_pattern = 'original_phrase'
        elif input_pattern == 2:
            self.input_pattern = 'translated_phrase_translated_gloss'
        elif input_pattern == 3:
            self.input_pattern = 'original_phrase_original_gloss'

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, item):
        amb = self.test_data.loc[item]['amb']
        phrase = self.test_data.loc[item]['phrase']

        ### (ori) phrase + (ori) sense
        ori_sense = 'sense_' + self.language
        if self.language == 'fa':
            prefix = ' عکس '
        elif self.language == 'it':
            prefix = 'Una foto di '

        if self.input_pattern == 'original_phrase':
            if self.language == 'en':
                sentence = 'A photo of ' + phrase
            else:
                sentence = prefix + phrase  # 不加任何sense

        elif self.input_pattern == 'translated_phrase_translated_gloss':
            if self.language == 'en':
                sentence = 'A photo of ' + phrase + ', ' + self.sense_sup.loc[item]['sense_en'].lower()
            else:
                sentence = 'A photo of ' + self.sense_sup.loc[item]['phrase_translation'].lower() + ', ' + self.sense_sup.loc[item]['sense_en'].lower()
        elif self.input_pattern == 'original_phrase_original_gloss':
            if self.language == 'en':
                sentence = 'A photo of ' + phrase + ', ' + self.sense_sup.loc[item]['sense_en'].lower()
            else:
                sentence = prefix + phrase + ', ' + self.sense_sup.loc[item][ori_sense].lower()

        img_names = self.test_data.loc[item]['candidate_imgs']
        img_paths = self.test_data.loc[item]['img_paths']
        candidate_images = torch.vstack([get_img_vec(path) for path in img_paths])
        gold_img = self.test_data.loc[item]['gold_img']

        return amb,sentence,img_names,candidate_images,gold_img

def collate_eval_fn(batch):
    amb,sentence,img_names,candidate_images,gold_img = zip(*batch)
    amb,sentence, img_names,candidate_images,gold_img = list(amb),list(sentence),list(img_names),list(candidate_images),list(gold_img)
    return amb,sentence,img_names,candidate_images,gold_img


def get_dataloader(args,data,mode,input_pattern=None):
    if mode =='train':
        mydataset = train_dataset(data, args=args, mode=mode)
        data_loader = DataLoader(mydataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,collate_fn=collate_train_fn)
    elif mode == 'test_en' or mode == 'test_fa' or mode == 'test_it':
        mydataset = test_dataset(data, args=args, mode=mode,input_pattern=input_pattern)
        data_loader = DataLoader(mydataset, batch_size=args.eval_batch_size, shuffle=False,num_workers=args.num_workers, pin_memory=True,collate_fn=collate_eval_fn)
    return data_loader

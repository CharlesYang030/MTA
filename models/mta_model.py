'''
 # Copyright
 # 2023/2/18
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CLIP import clip

class visual_encoder_withparas(nn.Module):
    def __init__(self,clip_model):
        super(visual_encoder_withparas, self).__init__()
        self.visual_encoder = clip_model.visual.float()
        self.dtype = torch.float16

    def forward(self, image):
        img_feat = self.visual_encoder(image.type(self.dtype))
        return img_feat.float()

class mtext_encoder_withparas(nn.Module):
    def __init__(self,clip_model):
        super(mtext_encoder_withparas, self).__init__()
        self.token_embedding = clip_model.token_embedding.float()
        self.positional_embedding = clip_model.positional_embedding.float()
        self.transformer = clip_model.transformer.float()
        self.ln_final = clip_model.ln_final.float()
        self.text_projection = clip_model.text_projection.float()
        self.dtype = torch.float16

    def forward(self,text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x.float()

class mtam(nn.Module):
    def __init__(self, args, data):
        super(mtam, self).__init__()
        self.args = args
        self.img2sen = data.imgid2senid
        self.device = args.device

        clip_version = 'ViT-L/14'

        clip_model, _ = clip.load(clip_version,device=args.device, jit=False)
        self.clip_tokenize = clip.tokenize

        self.embed = 768
        # create normal encoders
        self.vision_encoder = visual_encoder_withparas(clip_model)
        self.mtext_encoder = mtext_encoder_withparas(clip_model)

        self.momentum = 0.995
        # create momentum encoders
        self.vision_encoder_m = visual_encoder_withparas(clip_model)
        self.mtext_encoder_m = mtext_encoder_withparas(clip_model)

        self.model_pairs = [[self.vision_encoder, self.vision_encoder_m],
                            [self.mtext_encoder, self.mtext_encoder_m],
                            ]
        self.copy_params()

        self.image_queue_size = 110000
        self.text_queue_size = 35000
        # create queue
        self.register_buffer("image_queue", torch.randn(self.embed, self.image_queue_size))
        self.register_buffer("english_queue", torch.randn(self.embed, self.text_queue_size))
        self.register_buffer("concept_queue", torch.randn(self.embed, self.text_queue_size))

        self.register_buffer("image_idx_queue", torch.full((1, self.image_queue_size), -100))  # images idx
        self.register_buffer("mtext_idx_queue", torch.full((1, self.text_queue_size), -100))  # multilingual texts idx

        self.register_buffer("image_ptr_queue", torch.zeros(1, dtype=torch.long))
        self.register_buffer("mtext_ptr_queue", torch.zeros(1, dtype=torch.long))
        
        #self.temp = nn.Parameter(10*torch.ones([]))

    def forward(self, batch):
        ambiguous_word, word_language, sense_id, concepts, glosses, image_ids, image_vecs, image_labels = batch
        this_bs = len(ambiguous_word)  # 记录当前batch的长度

        # get gloss and concept
        en_gloss,fa_gloss,it_gloss,word_gloss,word_concept,en_gloss_pure,fa_gloss_pure,it_gloss_pure, \
            en_concept,fa_concept,it_concept= self.get_sentence(ambiguous_word,word_language, glosses, concepts, this_bs)

        # english text feats
        en_gloss_feats = self.mtext_encoder(self.clip_tokenize(en_gloss, truncate=True).to(self.device))
        en_gloss_feats = F.normalize(en_gloss_feats, dim=-1)

        en_gloss_pure_feats = self.mtext_encoder(self.clip_tokenize(en_gloss_pure, truncate=True).to(self.device))
        en_gloss_pure_feats = F.normalize(en_gloss_pure_feats, dim=-1)

        # farsi text feats
        fa_gloss_feats = self.mtext_encoder(self.clip_tokenize(fa_gloss, truncate=True).to(self.device))
        fa_gloss_feats = F.normalize(fa_gloss_feats, dim=-1)

        fa_gloss_pure_feats = self.mtext_encoder(self.clip_tokenize(fa_gloss_pure, truncate=True).to(self.device))
        fa_gloss_pure_feats = F.normalize(fa_gloss_pure_feats, dim=-1)

        # italian text feats
        it_gloss_feats = self.mtext_encoder(self.clip_tokenize(it_gloss, truncate=True).to(self.device))
        it_gloss_feats = F.normalize(it_gloss_feats, dim=-1)

        it_gloss_pure_feats = self.mtext_encoder(self.clip_tokenize(it_gloss_pure, truncate=True).to(self.device))
        it_gloss_pure_feats = F.normalize(it_gloss_pure_feats, dim=-1)

        # word_gloss text feats
        word_gloss_feats = self.mtext_encoder(self.clip_tokenize(word_gloss, truncate=True).to(self.device))
        word_gloss_feats = F.normalize(word_gloss_feats, dim=-1)

        # word_concept text feats
        word_concept_feats = self.mtext_encoder(self.clip_tokenize(word_concept, truncate=True).to(self.device))
        word_concept_feats = F.normalize(word_concept_feats, dim=-1)


        ###============== 1.(English)Text-Image Contrastive Learning ===================###
        en2i_labels = self.en2i_target(image_ids, image_labels, this_bs)

        total_feats = torch.cat([en_gloss_feats,fa_gloss_feats,it_gloss_feats,
                                 en_gloss_pure_feats,fa_gloss_pure_feats,it_gloss_pure_feats,
                                 word_gloss_feats,word_concept_feats],dim=0)
        total_labels = en2i_labels.repeat(8,1)

        with torch.no_grad():
            self._momentum_update()

            en_gloss_feats_m = self.mtext_encoder_m(self.clip_tokenize(en_gloss, truncate=True).to(self.device))
            en_gloss_feats_m = F.normalize(en_gloss_feats_m, dim=-1)
            # total_feats_m = en_gloss_feats_m.repeat(8,1)

            # image feats
            image_feats_m = self.vision_encoder_m(image_vecs.to(self.device))
            image_feats_m = F.normalize(image_feats_m, dim=-1)
            image_feats_m_all = torch.cat([image_feats_m.t(), self.image_queue.clone().detach()], dim=1)

            # sim_total2i_m = total_feats_m @ image_feats_m_all
            # sim_total2i_targets_m = alpha * F.softmax(sim_total2i_m, dim=1) + (1 - alpha) * total_labels

        sim_total2i = total_feats @ image_feats_m_all
        loss_total2i = -torch.sum(F.log_softmax(sim_total2i, dim=1) * total_labels, dim=1).mean()
        loss_it = loss_total2i

        ###============== 2.(no English)Text-(English)Text Contrastive Learning ===================###
        ot2en_labels = self.other2en_target(sense_id, this_bs)
        ot2engloss_labels = ot2en_labels.repeat(6, 1)
        with torch.no_grad():
            en_feats_m_all = torch.cat([en_gloss_feats_m.t(), self.english_queue.clone().detach()], dim=1)

        other_gloss_feats = torch.cat([fa_gloss_feats,it_gloss_feats,
                                       fa_gloss_pure_feats,it_gloss_pure_feats,
                                       word_gloss_feats,word_concept_feats],dim=0)
        sim_other2en = other_gloss_feats @ en_feats_m_all
        loss_oe = -torch.sum(F.log_softmax(sim_other2en, dim=1) * ot2engloss_labels, dim=1).mean()

        ot2enpure_labels = ot2en_labels.repeat(2, 1)
        with torch.no_grad():
            en_concept_feats_m = self.mtext_encoder_m(self.clip_tokenize(en_concept, truncate=True).to(self.device))
            en_concept_feats_m = F.normalize(en_concept_feats_m, dim=-1)
            en_concept_feats_m_all = torch.cat([en_concept_feats_m.t(), self.concept_queue.clone().detach()], dim=1)

        other_concepts = fa_concept + it_concept
        other_concepts_feats = self.mtext_encoder(self.clip_tokenize(other_concepts, truncate=True).to(self.device))
        other_concepts_feats = F.normalize(other_concepts_feats, dim=-1)

        sim_ohcp2encp = other_concepts_feats @ en_concept_feats_m_all
        loss_ohcp2encp = -torch.sum(F.log_softmax(sim_ohcp2encp, dim=1) * ot2enpure_labels, dim=1).mean()

        ### dequeue and enqueue
        self._dequeue_and_enqueue_image(image_feats_m, image_ids, this_bs)
        self._dequeue_and_enqueue_mtext(en_gloss_feats_m,en_concept_feats_m, sense_id, this_bs)

        loss = loss_it + (loss_oe + loss_ohcp2encp) / 2
        return loss

    def get_sentence(self, ambiguous_word,word_language, gloss, concepts, this_bs):
        en_gloss = []
        fa_gloss = []
        it_gloss = []
        en_concept = []
        fa_concept = []
        it_concept = []

        word_gloss = []
        word_concept = []
        en_gloss_pure = []
        fa_gloss_pure = []
        it_gloss_pure = []
        for i in range(this_bs):
            en_gloss.append('A photo of ' + concepts[i][0] + ', ' + gloss[i][0].lower())
            fa_gloss.append(' عکس ' + concepts[i][1] + ', ' + gloss[i][1])
            it_gloss.append('Una foto di ' + concepts[i][2] + ', ' + gloss[i][2].lower())
            en_concept.append(concepts[i][0])
            fa_concept.append(concepts[i][1])
            it_concept.append(concepts[i][2])

            en_gloss_pure.append('A photo of ' + gloss[i][0].lower())
            fa_gloss_pure.append(' عکس ' + gloss[i][1])
            it_gloss_pure.append('Una foto di ' + gloss[i][2].lower())
            if word_language[i] == 'en':
                word_gloss.append('A photo of ' + ambiguous_word[i] + ', ' + gloss[i][0].lower())
                word_concept.append('A photo of ' + ambiguous_word[i] + ', ' + concepts[i][0])
            elif word_language[i] == 'fa':
                word_gloss.append(' عکس ' + ambiguous_word[i] + ', ' + gloss[i][1])
                word_concept.append(' عکس ' + ambiguous_word[i] + ', ' + concepts[i][1])
            elif word_language[i] == 'it':
                word_gloss.append('Una foto di ' + ambiguous_word[i] + ', ' + gloss[i][2].lower())
                word_concept.append('Una foto di ' + ambiguous_word[i] + ', ' + concepts[i][2].lower())

        return en_gloss,fa_gloss,it_gloss,word_gloss,word_concept,en_gloss_pure,fa_gloss_pure,it_gloss_pure,en_concept,fa_concept,it_concept

    def en2i_target(self,image_ids,image_labels,this_bs):
        # 获得image all的idx
        en2i_ids = []
        for i in range(this_bs):
            en2i_ids += image_ids[i]
        en2i_ids = torch.from_numpy(np.array(en2i_ids)).unsqueeze(0).to(self.device)
        img_idx_all = torch.cat([en2i_ids, self.image_idx_queue.clone().detach()], dim=1)

        # 获得 english text to image 的 similarity targets
        en2i_targets = torch.zeros([this_bs, img_idx_all.size(1)]).float().to(self.device)
        for i in range(this_bs):
            num = np.array(image_labels[i]).sum()
            for j in range(num):
                x = torch.from_numpy(np.array([image_ids[i][j]])).unsqueeze(0).to(self.device)
                x = torch.eq(x,img_idx_all).squeeze(0).float()
                en2i_targets[i] = en2i_targets[i] + x
            # if en2i_targets[i].sum(0, keepdim=True) != 0:
            #     en2i_targets[i] = en2i_targets[i] / en2i_targets[i].sum(0, keepdim=True)
        return en2i_targets

    def other2en_target(self,sense_id,this_bs):
        # 获得english text all的idx
        oth_ids = torch.from_numpy(np.array(sense_id)).unsqueeze(0).to(self.device)
        en_idx_all = torch.cat([oth_ids, self.mtext_idx_queue.clone().detach()], dim=1)
        
        ids = torch.from_numpy(np.array(sense_id)).unsqueeze(0).to(self.device)

        ot2en_targets = torch.zeros([this_bs,en_idx_all.size(1)]).float().to(self.device)
        for i in range(this_bs):
            x = ids[0][i].unsqueeze(0)
            x = torch.eq(x,en_idx_all).squeeze(0).float()
            ot2en_targets[i] = ot2en_targets[i] + x
            # if ot2en_targets[i].sum(0, keepdim=True) != 0:
            #     ot2en_targets[i] = ot2en_targets[i] / ot2en_targets[i].sum(0, keepdim=True)
        return ot2en_targets


    @torch.no_grad()
    def copy_params(self):
        for idx, model_pair in enumerate(self.model_pairs):
            if idx == 0:  # idx == 0 表示VisualTransformer
                for param, param_m in zip(model_pair[0].visual_encoder.parameters(),
                                          model_pair[1].visual_encoder.parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient
            else:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for idx, model_pair in enumerate(self.model_pairs):
            # if idx == 0:  # idx == 0 表示VisualTransformer
            #     for param, param_m in zip(model_pair[0].visual_encoder.parameters(),
            #                               model_pair[1].visual_encoder.parameters()):
            #         param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            # else:
            #     for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
            #         param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            if idx != 0:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue_image(self, image_feats_m, image_ids, this_bs):
        # 获得当前batch的image ids
        img_ids = []
        for i in range(this_bs):
            img_ids += image_ids[i]
        img_ids = torch.from_numpy(np.array(img_ids)).unsqueeze(0).to(self.device)

        size = img_ids.size(1)
        ptr = int(self.image_ptr_queue)

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + size) <= self.image_queue_size:
            self.image_queue[:, ptr:ptr + size] = image_feats_m.T
            self.image_idx_queue[:, ptr:ptr + size] = img_ids
        else:
            gap = self.image_queue_size - ptr
            self.image_queue[:, ptr:self.image_queue_size] = image_feats_m[0:gap, :].T
            self.image_idx_queue[:, ptr:self.image_queue_size] = img_ids[:, 0:gap]

            self.image_queue[:, 0:size - gap] = image_feats_m[gap:gap + size, :].T
            self.image_idx_queue[:, 0:size - gap] = img_ids[:, gap:gap + size]

        ptr = (ptr + size) % self.image_queue_size  # move pointer
        self.image_ptr_queue[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_mtext(self, sentences_feats_m,en_concept_feats_m, sense_id, this_bs):
        # 获得当前batch的mtext ids
        sen_ids = torch.from_numpy(np.array(sense_id)).unsqueeze(0).to(self.device)

        size = sen_ids.size(1)
        # gather keys before updating queue
        ptr = int(self.mtext_ptr_queue)

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + size) <= self.text_queue_size:
            self.english_queue[:, ptr:ptr + size] = sentences_feats_m.T
            self.concept_queue[:, ptr:ptr + size] = en_concept_feats_m.T
            self.mtext_idx_queue[:, ptr:ptr + size] = sen_ids
        else:
            gap = self.text_queue_size - ptr
            self.english_queue[:, ptr:self.text_queue_size] = sentences_feats_m[0:gap, :].T
            self.concept_queue[:, ptr:self.text_queue_size] = en_concept_feats_m[0:gap, :].T
            self.mtext_idx_queue[:, ptr:self.text_queue_size] = sen_ids[:, 0:gap]

            self.english_queue[:, 0:size - gap] = sentences_feats_m[gap:gap + size, :].T
            self.concept_queue[:, 0:size - gap] = en_concept_feats_m[gap:gap + size, :].T
            self.mtext_idx_queue[:, 0:size - gap] = sen_ids[:, gap:gap + size]

        ptr = (ptr + size) % self.text_queue_size  # move pointer
        self.mtext_ptr_queue[0] = ptr



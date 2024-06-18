'''
从零开始的基于Transformer的中文->日语翻译机
√--使用  ×--放弃  ?--待加入
RMSNorm √
AdamW   ×
SwiGLU  √
分词     ?
梯度剪裁  √
dropout √
'''
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from time import sleep
import sentencepiece as spm

def get_attn_pad_mask(seq_q ,seq_k):
    '''
    :param q_len: batch_size
    :param k_len: batch_size
    :return:需要mask的部分为True
    '''
    batch_size,len_q = seq_q.size()
    batch_size,len_k = seq_k.size()
    pad_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_mask.expand(batch_size,len_q,len_k)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.linear = nn.Linear(d_model,tgt_vocab_size,bias=False).cuda()

    def forward(self,enc_inputs,dec_inputs):
        '''
        词向量是列向量
        :param enc_inputs:[batch_size,src_len]
        :param dec_inputs:[batch_size,tgt_len]
        :return:
        '''
        enc_outputs,enc_self_attns = self.encoder(enc_inputs)
        dec_outputs,dec_self_attns,dec_enc_attns = self.decoder(dec_inputs,enc_inputs,enc_outputs)
        linear_outputs = self.linear(dec_outputs)
        return linear_outputs.view(-1,linear_outputs.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size,d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self,enc_inputs):
        enc_outputs = self.src_emb(enc_inputs) #[batch_size,src_len,d_model]
        enc_outputs = self.pos_emb(enc_outputs) #[batch_size,src_len,d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attn = []
        for layer in self.layers:
            enc_outputs,enc_attn = layer(enc_outputs,enc_self_attn_mask)
            enc_self_attn.append(enc_attn)
        return enc_outputs,enc_self_attn

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn=MultiHeadAttention()
        self.ffn = SwiGLUFFN()

    def forward(self,enc_inputs,enc_self_attn_mask):
        enc_output,attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_output = self.ffn(enc_output)
        return enc_output,attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias=False)
        )
        # self.ln = nn.LayerNorm(d_model).cuda()
        self.ln = LlamaRMSNorm(d_model).cuda()

    def forward(self,inputs):

        outputs = self.fc(inputs)
        return self.ln(outputs+inputs)



class MultiHeadAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model,d_k*n_head,bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(d_v*n_head,d_model,bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input_Q,input_K,input_V,attn_mask):

        residual, batch_size=input_Q, input_Q.size(0)
        #  batch_size, seq_len, n_head * d_k -> batch_size, seq_len, n_head, d_k -> batch_size, n_head, seq_len, d_k
        Q = self.W_Q(input_Q).view(batch_size,-1,n_head,d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size,-1,n_head,d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size,-1,n_head,d_v).transpose(1,2)

        attn_mask=attn_mask.unsqueeze(1).repeat(1,n_head,1,1)
        context, attn = ScaledDotProductAttention()(Q,K,V,attn_mask) # b, n, seq_len, d_v
        context = context.transpose(1,2).reshape(batch_size,-1,n_head*d_v)
        output = self.fc(context) #b,n,dmodel
        # return nn.LayerNorm(d_model).cuda()(output+residual),attn
        return LlamaRMSNorm(d_model).cuda()(self.dropout(output) + residual), attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,Q,K,V,attn_mask):
        scores = torch.matmul(Q,K.transpose(-1,-2)) /np.sqrt(d_k)
        scores.masked_fill_(attn_mask,-1e6)
        scores = self.softmax(scores)
        context = torch.matmul(scores,V)
        return context,scores


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size,d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self,dec_inputs,enc_inputs,enc_output):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        dec_self_attn_mask = get_attn_pad_mask(dec_inputs,dec_inputs).cuda()
        dec_seq_mask = get_attn_seq_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_seq_mask+dec_self_attn_mask),0).cuda()
        enc_dec_attn_mask = get_attn_pad_mask(dec_inputs,enc_inputs).cuda()

        dec_self_attns,enc_dec_attns = [],[]
        for layer in self.layers:
            dec_outputs,dec_self_attn,enc_dec_attn = layer(enc_output,dec_outputs,dec_self_attn_mask,enc_dec_attn_mask)
            dec_self_attns.append(dec_self_attn)
            enc_dec_attns.append(enc_dec_attn)
        return dec_outputs,dec_self_attns,enc_dec_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer,self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.enc_dec_attn = MultiHeadAttention()
        self.ffn = SwiGLUFFN()

    def forward(self,enc_output,dec_input,dec_self_attn_mask,enc_dec_attn_mask):
        dec_outputs,dec_self_attn = self.dec_self_attn(dec_input,dec_input,dec_input,dec_self_attn_mask)
        dec_outputs,enc_dec_attn = self.enc_dec_attn(dec_outputs,enc_output,enc_output,enc_dec_attn_mask)
        dec_outputs = self.ffn(dec_outputs)
        return dec_outputs,dec_self_attn,enc_dec_attn

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)

class SwiGLUFFN(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.up = nn.Linear(d_model,d_ff,bias=False).cuda()
        self.down = nn.Linear(d_ff,d_model,bias=False).cuda()
        self.gate = nn.Linear(d_model,d_ff,bias=False).cuda()
        self.silu = nn.SiLU().cuda()
        self.ln = LlamaRMSNorm(d_model).cuda()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.ln(self.dropout(self.down(self.silu(self.gate(x))*self.up(x)))+x)

def get_attn_seq_mask(seq):
    sequence_mask = np.triu(np.ones([seq.size(0),seq.size(1),seq.size(1)]),k=1)
    sequence_mask = torch.from_numpy(sequence_mask).byte()
    return sequence_mask#b tgt_len tgt_len

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding,self).__init__()
        pe = torch.zeros(max_len,d_model)    #[src_len,d_model]
        pos = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) #两个一维向量无法对应位置相乘，所以加一维
        div = torch.exp(torch.arange(0,d_model,2,dtype=torch.float) * (-math.log(10000)/d_model) )
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        '''
        :param x: [batch_size,src_len,d_model]
        :return:[batch_size,src_len,d_model]
        '''
        x = x+self.pe[:,:x.size(1),:]
        return x

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def seqs2vec(seqs,vocab:str,coder:str):
    '''
    coder = 'enc'
            'dec_in'      就加开始符号
            'dec_out'   加停止符号
    '''
    sp = spm.SentencePieceProcessor(model_file=f'{vocab}.model')
    if coder == 'enc':
        vec = [sp.encode(seq) for seq in seqs]
    elif coder == 'dec_in':
        vec = [[2]+sp.encode(seq) for seq in seqs]
    elif coder == 'dec_out':
        vec = [sp.encode(seq)+[3] for seq in seqs]

    maxlen = max(len(seq) for seq in vec)
    vectors = []
    for i in vec:
        vector = torch.LongTensor(i)
        vector = F.pad(vector,(0,maxlen-len(vector)))
        vectors.append(vector)
    return torch.stack(vectors),maxlen


def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.linear(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == 3:
            terminal = True
        # print(next_word)
    return dec_input


if __name__ == "__main__":
    # sentences = [
    #     # enc_input           dec_input         dec_output
    #     ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    #     ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    # ]

    # seqs_input = ['你是笨蛋吗','我爱你','笨蛋我爱你','我爱笨蛋']
    # seqs_output = ['バカですか', '愛しています','バカ愛してます','私はバカが好きです']

    # # Transformer Parameters
    d_model = 512  # Embedding Size
    d_ff = 1024  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 8  # number of Encoder of Decoder Layer
    n_head = 6  # number of heads in Multi-Head Attention

    predict = int(input('预测扣1，训练扣0:'))
    if predict == False:
        src_vocab_size = spm.SentencePieceProcessor(model_file=f'ch_model.model').vocab_size()
        tgt_vocab_size = spm.SentencePieceProcessor(model_file=f'jp_model.model').vocab_size()
        with open('yourname_ch.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            seqs_input = [i.rstrip('\n') for i in lines]
        with open('yourname_jp.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            seqs_output = [i.rstrip('\n') for i in lines]

        enc_inputs, src_len = seqs2vec(seqs_input, 'ch_model','enc')
        dec_inputs, tgt_len = seqs2vec(seqs_output, 'jp_model', 'dec_in')
        dec_outputs, tgt_len = seqs2vec(seqs_output, 'jp_model', 'dec_out')

        train_on_pretrain_model = int(input('继续训练扣1，从头开始扣0:'))
        if train_on_pretrain_model == 1:
        #在已有的模型上继续训练
            model = torch.load('model_parameters1.pth')
        else:
        #从头开始训练
            model = Transformer().cuda()

        #只是用前100000条数据减少数据量，跑快点
        # enc_inputs, dec_inputs, dec_outputs = enc_inputs[300000:], dec_inputs[300000:], dec_outputs[300000:]
        loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 80, True)
        # model = Transformer().cuda()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        torch.backends.cudnn.benchmark=True

        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
        # optimizer = optim.AdamW(model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,100000,1e-5)
        iter_num = int(input('训练轮数:'))
        i = 0
        for epoch in range(iter_num):
            for enc_inputs, dec_inputs, dec_outputs in loader:
                '''
                enc_inputs: [batch_size, src_len]
                dec_inputs: [batch_size, tgt_len]
                dec_outputs: [batch_size, tgt_len]
                '''
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
                # outputs: [batch_size * tgt_len, tgt_vocab_size]
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                loss = criterion(outputs, dec_outputs.view(-1)) / 100
                # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                loss.backward()
                i += 1
                if i % 100 == 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if i % 1000 == 0:
                    torch.save(model, 'model_parameters1.pth')
                    print(f'已训练{i}步,loss={loss * 100:.6f}')


# Test
#     enc_inputs, _, _ = next(iter(loader))
    else:
        model = torch.load('model_parameters1.pth')
        model.eval()
        path = ChromeDriverManager().install()
        url = 'https://fanyi.youdao.com/#/'
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(service=Service(path), options=options)  # Here
        driver.get(url)
        sleep(1)
        driver.implicitly_wait(30)
        sp = spm.SentencePieceProcessor(model_file='jp_model.model')
        seq = input("输入你要翻译的句子(输入0就退出):")
        while seq!='0':
            enc_inputs,_ = seqs2vec([seq],'ch_model','enc')
            # print(enc_inputs)
            enc_inputs = enc_inputs.cuda()
            for i in range(len(enc_inputs)):
                greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=2)
                predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
                predict = predict.data.max(1, keepdim=True)[1]
                # print(enc_inputs[i], '->', [idx2word[f'{n.item()}'] for n in predict.squeeze()])
            ans = [n.item() for n in predict.squeeze()]
            ans = sp.decode(ans)
            print(ans)


            # 谷歌浏览器，启动！
            driver.find_element(By.ID, 'js_fanyi_input').clear()
            sleep(0.2)
            driver.find_element(By.ID, 'js_fanyi_input').send_keys(ans)
            sleep(1.5)
            print(driver.find_element(By.ID, 'js_fanyi_output_resultOutput').text)
            seq = input("输入你要翻译的句子(输入0就退出):")
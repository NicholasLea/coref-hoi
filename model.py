import torch
import torch.nn as nn
from transformers import BertModel
import util
import logging
from collections import Iterable
import numpy as np
import torch.nn.init as init
import higher_order as ho


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device

        # ?genres是怎么带进去的，后面了再看看
        self.num_genres = num_genres if num_genres else len(config['genres'])  # genres是在数据处理的过程中就带进去的，后面直接用
        self.max_seg_len = config['max_segment_len']  # 128
        self.max_span_width = config['max_span_width']  # 30
        assert config['loss_type'] in ['marginalized', 'hinge']
        # 关于higher_order，论文里提到了4中方法，attended antecedent, entity equalization, span clustering, and cluster merging
        # 但是在配置文件里还有max_antecedent这一种，共5种
        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']  # Higher-order is in slow fine-grained scoring

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config[
            'use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config[
            'model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                  output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'],
                                                    [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if \
        config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config[
            'use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                               output_size=1) if config['fine_grained'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config[
                                                                                                          'coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config[
                                                                                          'higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'],
                                                 [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if \
        config['higher_order'] == 'cluster_merging' else None

        self.update_steps = 0  # Internal use for debug
        self.debug = True

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                 is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        print('input_ids.shape:', input_ids.shape)
        print('input_mask.shape:', input_mask.shape)
        print('speaker_ids.shape:', speaker_ids.shape)
        print('sentence_len.shape:', sentence_len.shape)
        print('genre.shape:', genre.shape)
        print('sentence_map.shape:', sentence_map.shape)

        print('---')
        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True
        # 这里是有gold_mention_cluster_map的，它应该是在之前被封装进来了
        # print('gold_mention_cluster_map:', gold_mention_cluster_map)
        # print('gold_starts:', gold_starts)
        # print('gold_ends:', gold_ends)
        # print('do_loss:', do_loss)

        # Get token emb
        # 这种解包方式解出来的结果是两个字符串'last_hidden_state', 'pooler_output',需要用最新的方式
        # mention_doc, _ = self.bert(input_ids, attention_mask=input_mask)  # [num seg, num max tokens, emb size]
        # 改成下面这种就好了
        # num seg就是有几个对话片段（句子）
        # get the embedding of each tokens, shape: [sent num, input length, embeding size]
        mention_doc = self.bert(input_ids, attention_mask=input_mask)[
            0]  # [num seg 几个片段, num max tokens 128, emb size 768]
        input_mask = input_mask.to(torch.bool)  # shape: [sent num, input length]
        # mention_doc返回的是包含mask的值的。
        # print('mention_doc.shape bef',mention_doc.shape)

        # 通过加总input_mask里的值，可以找到一共有多少个词，这和sentence_map的长度是对应的
        # print('input_mask.sum(axis=1)',input_mask.sum(axis=1))
        # print('sentence_map.shape',sentence_map.shape)

        # input mask的值是True\False，比如[[True,True],[True,False]]，那么选择后就留下来三个True，原来
        # [2,2,768]的结构就会变成[3,768]
        # print('speaker_ids.shape',speaker_ids.shape) 这里也是包含了mask_input
        # print('sentence_map',sentence_map)
        '''after using bert, we don't need input mask, so we eliminate the padding elements
        Notice: below two lines will decrease two-dimension to one-dimension. Such as,
        Input_mask=[[1,0],[1,1]] # shape [2,2]
        mention_doc = [[Vector0,Vector1],[Vector2,Vector3]] #shape [2,2,768]
        Code: mention_doc = mention_doc[input_mask] 
        mention_doc = [Vector0,Vector2,Vector3] #  [3,768]
        '''
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]  # one vector
        # print('speaker_ids',speaker_ids)
        # 一个speaker_id会对应多个sentence_map，因为它会说多句话。但是不会反过来，因为一个句子不能由多个人说
        # 这里进行的mask只是为了使用bert。
        # print('input_mask.shape',input_mask.shape)
        # print('mention_doc.shape',mention_doc.shape)
        # print('speaker_ids.shape',speaker_ids.shape)
        # first shape is number of words. We can also use sentence_map.shape[0]
        num_words = mention_doc.shape[0]
        print('num_words:', num_words)  # token的数量
        # print('sentence_map:',sentence_map.shape[0])

        # Get candidate span
        # print('sentence_map',sentence_map)
        # print('len(sentence_map)',len(sentence_map))
        sentence_indices = sentence_map  # [num tokens] 长度等于num tokens，是每个句子的索引号，从0开始

        # 初始化了一个矩阵以存放candi的起始位置。首先每个词都可能成为candi，所以第一维度的长是词汇数，
        # 其次candi的长度一共只能是max_span_width(30)，所以第二位是30。这里第二维度重复了30次，只是为了配个end使用
        candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1, self.max_span_width)
        # max_span_width在设置里是30。 shape: [num word, 30]
        # print('candidate_starts',candidate_starts)
        # print('candidate_starts.shape',candidate_starts.shape)
        ''' 它的格式类似于[  0,   1,   2,  ...,  27,  28,  29]，枚举了每一个可能的起始位置
            candidate_starts tensor([[  0,   0,   0,  ...,   0,   0,   0],
                                    [  1,   1,   1,  ...,   1,   1,   1],
                                    [  2,   2,   2,  ...,   2,   2,   2],
                                    ...,
                                    [226, 226, 226,  ..., 226, 226, 226],
                                    [227, 227, 227,  ..., 227, 227, 227],
                                    [228, 228, 228,  ..., 228, 228, 228]], device='cuda:0')
        '''
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        # print('candidate_ends',candidate_ends)
        '''candidate_ends tensor([[  0,   1,   2,  ...,  27,  28,  29],
                                  [  1,   2,   3,  ...,  28,  29,  30],
                                  [  2,   3,   4,  ...,  29,  30,  31],
                                  ...,
                                  [221, 222, 223,  ..., 248, 249, 250],
                                  [222, 223, 224,  ..., 249, 250, 251],
                                  [223, 224, 225,  ..., 250, 251, 252]], device='cuda:0')
        '''
        # 找出对应的句子的idx
        # pick up the sentence id of candidate_starts and candidate_ends
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        # torch.min to avoid beyond the list
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]

        # candidate_start_sent_idx == candidate_end_sent_idx要求必须在一个句子里
        # candidate_start_sent_idx == candidate_end_sent_idx requires the span is in the same sentence
        # candidate_ends < num_words 是为了过滤掉candidate_ends里那些idx超过了最大token的。这个操作方法挺好的，值得学习

        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        print('candidate_mask.shape', candidate_mask.shape)

        # mask掉一遍。同时会变成一维的
        # notice: candidate_starts, candidate_ends will become a vector with shape [num valid candidates]
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[
            candidate_mask]  # [num valid candidates]
        # print('candidate_starts',candidate_starts)
        # print('candidate_starts.shape',candidate_starts.shape)
        # print('candidate_ends',candidate_ends)
        num_candidates = candidate_starts.shape[0]
        print('num_candidates', num_candidates)
        # 到此为止，我们得到了所有可能的candi的起始和结束位置
        # 感觉上述代码写的还是不错的

        # Get candidate labels
        if do_loss:  # 回忆下如果有gold_mention的信息，则do_loss。这部分都是针对gold的处理。
            # print('gold_starts',gold_starts)
            # print('torch.unsqueeze(gold_starts, 1)',torch.unsqueeze(gold_starts, 1))
            # print('candidate_starts',candidate_starts)
            # print('torch.unsqueeze(candidate_starts, 0)',torch.unsqueeze(candidate_starts, 0))
            # torch.unsqueeze(gold_starts, 1)会变成[[30],[32],...,[49]]这样
            # torch.unsqueeze(candidate_starts, 0)) 会变成 [[   0,    0,    0,  ..., 1148, 1149, 1150]]
            # print('torch.unsqueeze(gold_starts, 1)',torch.unsqueeze(gold_starts, 1).shape)
            # print('torch.unsqueeze(candidate_starts, 0)',torch.unsqueeze(candidate_starts, 0).shape)
            ''' gold_starts and candidate_starts are both vectors. (torch.unsqueeze(gold_starts, 1) is [gold num, 1],
            torch.unsqueeze(candidate_starts, 0) is [1, candidate num], so the same_start is [gold num, candidate num].
            '''
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            # 这个same_start蛮有趣的，它是[15, 1]和[1, 1498]比较，会变成[15,1498]。它的意义可以解释为，15个gold_mention里的
            # 每一个，它对应1498个候选里的哪一个（或者任何一个都不对应）
            # print('same_start',same_start)
            # print('same_start.shape',same_start.shape)

            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            # same_end与same_start同理
            # print('same_start.sum()',same_start.sum())
            # print('same_end.sum()',same_end.sum())
            # print('same_end.shape',same_end.shape)

            same_span = (same_start & same_end).to(torch.long)  # [gold num, candidate num]
            ''' 判断是否是same_span。注意same_span和same_start\same_end的shape是一样的，为[gold_span num, candi_num]，这是个
                对应了所有gold_span 和candi span的二维矩阵 
            '''
            # print('---')
            # print('same_span',same_span)
            # print('same_span[:, 0:10]',same_span[:, 0:10])
            # print('same_span.shape',same_span.shape)

            # print('torch.unsqueeze(gold_mention_cluster_map, 0)',torch.unsqueeze(gold_mention_cluster_map, 0).shape)
            # 这里的相乘是[1, gold_mention num]*[gold_mention num, candi num]，通过这种方式，就得到了每一个candi的label(aka 所属的cluster id)
            # [1, gold_mention num]*[gold_mention num, candidate num]= [1, candidate num] so we get the label of each candidate span
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            # print('gold_mention_cluster_map',gold_mention_cluster_map)
            # print('candidate_labels[1]',candidate_labels[0,:10])
            # print('candidate_labels',candidate_labels)
            # print('candidate_labels.shape',candidate_labels.shape)
            # 它这里的逻辑就是用gold_mention_cluster_map分别去乘以candi num个向量（每个代表这个candi与所有gold span是否相同），
            # 因为最多只会有一个相同，所得相加就会得到这个candi所对应的gold span的cluster id.这个矩阵乘法还不是那么容易的
            # 注意这里就会给所有的candi标记上label，因为candi和gold有穷举所有的对应关系

            # 这里再把[1,candi数]的这个1给去掉，得到[candi数]
            # [1, candidate num]-> [candidate num]; non-gold span has label 0
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long),
                                             0)  # [num candidates]; non-gold span has label 0
            # print('candidate_labels.shape',candidate_labels.shape)
            # 这里就相当于用gold span把所有的candi span给标记上了！
            # 这里这一通操作，还略复杂啊，它居然不在数据处理的步骤做，全都放在了模型里面，真不知道好还是不好

        # Get span embedding
        # print('mention_doc.shape',mention_doc.shape)
        # print('candidate_starts.shape',candidate_starts.shape)
        # print('candidate_ends.shape',candidate_ends.shape)
        # 找出candi start和end对应的emb
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]
        if conf['use_features']:
            # 这里也是有use_features的
            # print('use_features.')
            # print('candidate_starts.shape',candidate_starts.shape)
            # print('candidate_ends.shape',candidate_ends.shape)
            # 这里就是存储了各个candi的长度，是一个一维的list(torch)
            candidate_width_idx = candidate_ends - candidate_starts
            # print('candidate_width_idx.shape',candidate_width_idx.shape)
            # 这里就是新建了一个emd layer，对长度这个数字进行了emdding，然后dropout了
            #  self.emb_span_width is a embedding layer
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            # print('candidate_width_emb.shape 1',candidate_width_emb.shape)
            candidate_width_emb = self.dropout(candidate_width_emb)
            # print('candidate_width_emb.shape 2',candidate_width_emb.shape)
            candidate_emb_list.append(candidate_width_emb)  # 最后加到embedding的列表里
            # 注意这里说use_feature，实际上只用了长度这个feature，可能还是可以用更多feature的！

        # Use attended head or avg token
        # 初始化了一个矩阵，第一维度是num_candidates（每一个candi都可能是mention），第二位是[0,1,...,num_words]（标出具体的idx）。
        # candidate_tokens: [num_candidates, num_words]
        # we use a matrix [num_candidates, num_words] to represent the candidate tokens. It includes all possible tokens.
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        # 在axis=1上重复，于是num_candidates会成为第一个维度的值
        print('candidate_tokens.shape', candidate_tokens.shape)
        # print('candidate_tokens',candidate_tokens)
        # print('candidate_starts',candidate_starts)
        # print('torch.unsqueeze(candidate_starts, 1)',torch.unsqueeze(candidate_starts, 1).shape)
        # print('candidate_tokens.shape',candidate_tokens.shape)
        # torch.unsqueeze(candidate_starts, 1): [num_candidates, 1]。进行第一个比较后变成了[num_candidates, num_words]
        # print('(candidate_tokens >= torch.unsqueeze(candidate_starts, 1))',(candidate_tokens >= torch.unsqueeze(candidate_starts, 1)))

        """ 解析：
          但是要注意只有在candidate_starts是[0,1,2,3...]的时候才成立，但是candidate_starts不是这样的。所以它不是严格的长方形矩阵的对角线掩码
          这里对start和end掩码，就是对所以不可能的都掩掉
               """
        # for our tokens with spans, it should from starting position and to ending positon
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (
                    candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        # print('candidate_tokens_mask[100]',candidate_tokens_mask[100])
        # print('candidate_starts[100]', candidate_starts[100])
        # print('candidate_ends[100]', candidate_ends[100])
        # print('candidate_tokens_mask.shape',candidate_tokens_mask.shape)
        # 前面标好了candi的起始和结束位置，这里就是把所有的包含起始和中间位置的token的位置都标出来

        if conf['model_heads']:
            # print('model_heads') #这里是有model_heads的
            # print('mention_doc.shape', mention_doc.shape)
            # print('self.mention_token_attn(mention_doc).shape', self.mention_token_attn(mention_doc).shape)
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)  # [num_words]
            # print('token_attn.shape', token_attn.shape)
            # mention_token_attn就是对应一个线性转换，nn.Linear，把768维度的向量转化为一个单一的数字。
            # 这里是得到一个长度为 num_words的向量，它是把mention_doc经过了ffnn得到的，
            # 可以认为是整个mention所在的doc的所有token的representation。接下来就用于candi span中间的token的表示
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
            # 如果没有attention，那么就是用平均。注意这里的attention就是用一个线性层把它给加权起来，权重的话就自己计算

        # to(torch.float)转化为了0.0和1.0。log(1)等于0，而log(0)等于负无穷。它这是一种mask的方式。
        # to(torch.float) makes False to 0 and True to 1. log(1)=0，log(0)=-inf. -inf add anything equals -inf.
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        # print('candidate_tokens_mask.to(torch.float)',candidate_tokens_mask.to(torch.float))
        # print('torch.log(candidate_tokens_mask.to(torch.float))',torch.log(candidate_tokens_mask.to(torch.float)))
        # print('torch.log(candidate_tokens_mask.to(torch.float)).shape',torch.log(candidate_tokens_mask.to(torch.float)).shape)
        # print('torch.unsqueeze(token_attn, 0).shape',torch.unsqueeze(token_attn, 0).shape)
        # print('candidate_tokens_attn_raw.shape',candidate_tokens_attn_raw.shape)
        # print('candidate_tokens_attn_raw',candidate_tokens_attn_raw)
        # 负无穷在softmax后会变成0，会自动不参与其他的softmax。如[1,1,-inf]会变成[0.5,0.5,0]
        # -inf will be 0 after softmax, will not influence others. Such as, softmax([1,1,-inf])=[0.5,0.5,0]
        # So, softmax(log(mask_matrix)+value_matrix) is a method for masking.
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        # print('candidate_tokens_attn.shape',candidate_tokens_attn.shape)
        # print('mention_doc.shape',mention_doc.shape)
        # 观察第100个
        # print('candidate_tokens_mask[100]',candidate_tokens_mask[100])
        # print('candidate_starts[100]', candidate_starts[100])
        # print('candidate_ends[100]', candidate_ends[100])
        # print('candidate_tokens_attn[100]', candidate_tokens_attn[100])
        # !这一段就是把candi span的从起始到结束的每一个token的embed所计算的embed scalar给拿出来了

        # print('candidate_tokens_attn',candidate_tokens_attn)
        # print('mention_doc',mention_doc)
        # print('candidate_tokens_attn',candidate_tokens_attn)
        # [num_candidates, num_words] * [num_words, 768] = [num_candidates, 768]
        # candidate_tokens_attn是 is attention weight，then multiply embbing from mention_doc，get candidates' attention-embeding
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        #
        # ！candidate_tokens_attn是attention weight，然后乘以mention_doc的emb，就得到了attention后的 candi的emb
        # print('head_attn_emb.shape',head_attn_emb.shape)
        # !前面得到了 candi span从start到end的一个scalar，这里再乘以mention_doc，这样就得到了每一个candi的emb(768维).
        # 前面的scalar就相当于是权重，不过也是算出来的

        candidate_emb_list.append(head_attn_emb)
        # 目前为止，这里有candi span的start emb\end emb\length emb\attn_emb
        # until now, we have start emb(768)\end emb(768)\length emb(20)\attn_emb(768).
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size(2324)]
        # 把这四个emb给cat起来
        # print('len(candidate_emb_list)', len(candidate_emb_list))
        # print('candidate_emb_list[0].shape', candidate_emb_list[0].shape) # 768
        # print('candidate_emb_list[1].shape', candidate_emb_list[1].shape) # 768
        # print('candidate_emb_list[2].shape', candidate_emb_list[2].shape) # 20
        # print('candidate_emb_list[3].shape', candidate_emb_list[3].shape) # 768
        # print('candidate_span_emb.shape',candidate_span_emb.shape) # 2324

        # Get span score 注意是candi span。
        # print("config['ffnn_size']", config['ffnn_size'])
        # print("[config['ffnn_size']]", [config['ffnn_size']])
        # print("config['ffnn_depth']",config['ffnn_depth'])
        # print("[config['ffnn_size']] * config['ffnn_depth']", [config['ffnn_size']] * config['ffnn_depth'])
        # print("self.span_emb_score_ffnn", self.span_emb_score_ffnn)
        ''' self.span_emb_score_ffnn是一个全连接层，如下，这里这个封装的函数还是挺好用的
            Sequential(
              (0): Linear(in_features=2324, out_features=3000, bias=True)
              (1): ReLU()
              (2): Dropout(p=0.3, inplace=False)
              (3): Linear(in_features=3000, out_features=1, bias=True)
            )
            这个函数对candidate_span_emb进行了全连接操作，把[1447, 2324]这种变成了[1447]
        '''
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        # print('candidate_mention_scores.shape', candidate_mention_scores.shape)
        # 这里就是算了一个分数出来，基于前面的emb

        if conf['use_width_prior']:
            # print('use_width_prior') #bert的配置使用了这个

            '''
            self.span_width_score_ffnn: Sequential(
              (0): Linear(in_features=20, out_features=3000, bias=True)
              (1): ReLU()
              (2): Dropout(p=0.3, inplace=False)
              (3): Linear(in_features=3000, out_features=1, bias=True)
            )
            '''
            # print('self.span_width_score_ffnn',self.span_width_score_ffnn)
            # print('self.emb_span_width_prior.weight', self.emb_span_width_prior.weight.shape)
            # print('self.span_width_score_ffnn(self.emb_span_width_prior.weight).shape', \
            # self.span_width_score_ffnn(self.emb_span_width_prior.weight).shape)
            # 这里就是对于宽度为30的每个位置上的数字，都算出来它的embed向量，然后加上去一个全连接层，算出来一个分数，也就是得到一个
            # 向量[30]。这么做我觉得是为了使用这个分数，同时又让它是可训练的。不过这里的代码封装的真是简介。咔咔复杂。
            ''' Similar with before, self.emb_span_width_prior is an embedding layer, so 
            self.emb_span_width_prior.weight is an matrix. self.span_width_score_ffnn is as below:
            self.span_width_score_ffnn: Sequential(
              (0): Linear(in_features=20, out_features=3000, bias=True)
              (1): ReLU()
              (2): Dropout(p=0.3, inplace=False)
              (3): Linear(in_features=3000, out_features=1, bias=True)
            )
            We get [max_span_width, 1].
            After torch.squeeze, we get [max_span_width]
            '''
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            # print('width_score.shape',width_score.shape)

            # candidate_width_idx是candi的长度，这个向量的长度是几百几千。width_score是一个30长度的向量。
            # 最终width_score[candidate_width_idx]的长度和candidate_width_idx是一样的
            ''' candidate_width_idx is [num_candidates], width_score is [max_span_width].
            This is a typical out_vector(/matrix)[in_vector(/matrix)], pick value from out_vector
            from position in in_vector.
            candidate_width_score is [max_span_width]
            '''
            candidate_width_score = width_score[candidate_width_idx]
            # print('candidate_width_idx.shape',candidate_width_idx.shape)
            # print('candidate_width_score.shape',candidate_width_score.shape)
            # 把长度的分值叠加到candidate_mention_scores上。注意这里都是先算好所有分值之后叠加上去，而不是在向量上就开始叠加然后
            # 一起计算。我觉得这样是合理的，因为向量不能乱加，他们不在一个空间内。
            # 这样就找出来了每一个candi的宽度的embed

            # 观察第100个
            # print('candidate_starts[100]', candidate_starts[100])
            # print('candidate_ends[100]', candidate_ends[100])
            # print('candidate_width_idx[100]', candidate_width_idx[100])
            # print('candidate_width_score[100]', candidate_width_score[100])

            # add with score to mention score
            candidate_mention_scores += candidate_width_score
            # 这里是把分数加上去了，这个没关系，只要不是向量直接加上去就行
            # 注意这里的分数计算其实有点不准，比如s=13,e=21,长度应该是9，但是这里直接是21-13=8

        # Extract top spans 以上得到了所有span的score，那么接下来就通过这些score降序选取top的span
        ''' argsort will return the ranking index of numbers, for example,
        argsort([3,9,6], descending=True) will return [2, 0, 1] which means 3 is the third, 9 is the first, 
        and 6 is the second after descending ranking. 
        len == num_candidates
        '''
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        # 返回降序排列的序号，如[0,907]代表[排位0，排位907]
        # print('candidate_idx_sorted_by_score', candidate_idx_sorted_by_score)
        # print('len(candidate_idx_sorted_by_score)', len(candidate_idx_sorted_by_score))
        # put numbers to CPU
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        # min(max spans, span ration * number of words)
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        print('num_top_spans', num_top_spans)

        # 这里就是给出排序，给出所有的candi的开始和结束，然后返回开始和结束列表对应的id。
        # 这里的逻辑也挺复杂的，吭哧吭哧的，有点当初做那个java项目的感觉了。这里应该主要是考虑了避免交叉，后续再细看
        # Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop
        # len == num_top_spans
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                                   candidate_ends_cpu, num_top_spans)

        # print('candidate_starts_cpu', candidate_starts_cpu)
        # print('candidate_ends_cpu', candidate_ends_cpu)
        # print('selected_idx_cpu', selected_idx_cpu)

        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        # starts/ends/emb/clusterids/mention_scores of top spans
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        # print('top_span_cluster_ids', top_span_cluster_ids)
        top_span_mention_scores = candidate_mention_scores[selected_idx]
        '''
        Until there, we have select top number spans by score (candidate_mention_scores). Next, we need select 
        top candidate antecedents 
        '''

        # Coarse (粗糙的) pruning on each mention's antecedents
        # print('num_top_spans', num_top_spans)
        # print('max_top_antecedents', conf['max_top_antecedents'])

        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])  # 50
        # print('max_top_antecedents', max_top_antecedents)
        top_span_range = torch.arange(0, num_top_spans, device=device)
        '''antecedent_offsets tensor([[  0,  -1,  -2,  ..., -62, -63, -64],
                                      [  1,   0,  -1,  ..., -61, -62, -63],
                                      [  2,   1,   0,  ..., -60, -61, -62],
                                      ...,
                                      [ 62,  61,  60,  ...,   0,  -1,  -2],
                                      [ 63,  62,  61,  ...,   1,   0,  -1],
                                      [ 64,  63,  62,  ...,   2,   1,   0]], device='cuda:0')
          which means the antecedent can only be the one before it.
        '''
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        # print('antecedent_offsets', antecedent_offsets)

        # 这里的antecedent_offsets就是为了计算antecedent_mask
        # choose left bottom corner (not including diagonal)
        antecedent_mask = (antecedent_offsets >= 1)  # 取左下三角（不含对角线）
        # print('antecedent_mask', antecedent_mask)
        # print('antecedent_mask.shape', antecedent_mask.shape)
        # 顾名思义，就是一对对的分值。这个写法很好呀！
        ''' compute each pairwaise socre sum.
            Compute pairwise value sum: unsqueeze(Value_vector, 1) + unsqueeze(Value_vector, 0)
            Compute pairwise position offset: unsqueeze(index_vector, 1) + unsqueeze(index_vector, 0), where
              index_vector = [0,1,2,...,N]
        '''
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(
            top_span_mention_scores, 0)
        # 计算两两配对分，这种写法在本代码中非常常见
        # print('pairwise_mention_score_sum.shape', pairwise_mention_score_sum.shape)

        # print('self.coarse_bilinear', self.coarse_bilinear)
        '''self.coarse_bilinear：
            self.coarse_bilinear Linear(in_features=2324, out_features=2324, bias=True)
            A simple linaer transformation.
        '''
        # print('top_span_emb.shape', top_span_emb.shape)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))  # 进行一个全连接层的转换
        # print('source_span_emb.shape', source_span_emb.shape)
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        # print('target_span_emb.shape', target_span_emb.shape)
        # 得到两两配对的分数
        # Dot product between source_span_emb and target_span_emb
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        # print('pairwise_coref_scores.shape', pairwise_coref_scores.shape)
        # pairwise_mention_score_sum 是 mention score两两配对相加；pairwise_coref_scores是emb两两点积

        # 把这两个加起来了。这两个区别好像不大，都是从 candidate_span_emb 起源的，pairwise_mention_score_sum是从candidate_span_emb
        # 变成分数，然后进行top操作。而pairwise_coref_scores是top选出来emb后，再算内积，pairwise_coref_scores可以认为是算coref的情况，
        # 而pairwise_mention_score_sum是在算mention的分数。最后得到的还是一个装满分数的list
        # Here, 看下paper里的公式
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        # 把上面两个加起来
        # print('pairwise_mention_score_sum.shape', pairwise_mention_score_sum.shape)
        # print('pairwise_coref_scores.shape', pairwise_coref_scores.shape)
        # print('pairwise_fast_scores.shape', pairwise_fast_scores.shape)
        # print('pairwise_fast_scores', pairwise_fast_scores)

        # antecedent_mask 是False的部分，会变成-inf
        # log(0)=-inf
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        # 把不可能的配对关系都屏蔽掉（不会出现A-B和B-A这种重复的情况）
        # print('pairwise_fast_scores', pairwise_fast_scores)
        # print('pairwise_fast_scores.shape', pairwise_fast_scores.shape)
        if conf['use_distance_prior']:  # 这里使用了
            # print('use_distance_prior')
            # print('self.emb_antecedent_distance_prior', self.emb_antecedent_distance_prior) #Embedding(10, 20) 这里是10是因为有10个区间段
            # print('self.emb_antecedent_distance_prior.weight.shape', self.emb_antecedent_distance_prior.weight.shape) # [10,20]
            '''self.emb_antecedent_distance_prior is Embedding(10, 20). The 10 is because we will use 10 distance classes.
            distance_score shape: [10]
            '''
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            # print('distance_score.shape', distance_score.shape) # [10] 这里和上面类似，distance一共有10种可能的取值
            # print('antecedent_offsets.shape', antecedent_offsets.shape)
            # print('antecedent_offsets',antecedent_offsets)
            ''' antecedent_offsets:
            [[   0,   -1,   -2,  ..., -194, -195, -196],
              [   1,    0,   -1,  ..., -193, -194, -195],
              [   2,    1,    0,  ..., -192, -193, -194],
              ...,
              [ 194,  193,  192,  ...,    0,   -1,   -2],
              [ 195,  194,  193,  ...,    1,    0,   -1],
              [ 196,  195,  194,  ...,    2,    1,    0]]

              bucketed_distance 
             [[0, 0, 0,  ..., 0, 0, 0],
              [1, 0, 0,  ..., 0, 0, 0],
              [2, 1, 0,  ..., 0, 0, 0],
              ...,
              [9, 9, 9,  ..., 0, 0, 0],
              [9, 9, 9,  ..., 1, 0, 0],
              [9, 9, 9,  ..., 2, 1, 0]]
            '''
            # 待 bucket_distance这个函数后面有空了看看。不过顾名思义，就是把距离给分桶，分成10个等级
            # antecedent_offsets还用来计算candi span两两距离，这种距离是按照分值排序后产生的距离
            # 注意offset本身就是距离了，bucketed_distance进一步对这个距离分桶了
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            # print('bucketed_distance',bucketed_distance)

            # print('bucketed_distance', bucketed_distance)
            # print('bucketed_distance.shape', bucketed_distance.shape)
            antecedent_distance_score = distance_score[bucketed_distance]  # 之后从这10个等级里去找对应的分数
            pairwise_fast_scores += antecedent_distance_score  # 把距离这个指标给加进去

        '''  
        pairwise_fast_scores: [num_top_spans, num_top_spans]
        top_pairwise_fast_scores: [num_top_spans, max_top_antecedents]
        '''
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        # 因为max_top_antecedents是50，所以会把[89,89]变为[89,50]
        # print('pairwise_fast_scores.shape',pairwise_fast_scores.shape)
        # print('top_pairwise_fast_scores',top_pairwise_fast_scores)
        # print('top_pairwise_fast_scores.shape',top_pairwise_fast_scores.shape)
        # print('top_antecedent_idx',top_antecedent_idx) #这个top_antecedent_idx是越小越好
        # print('top_antecedent_idx.shape',top_antecedent_idx.shape)

        # batch_select 这个还需要再看看!
        # print('top_antecedent_mask.shape bef',antecedent_mask.shape)
        # 这里就是把antecedent_mask从[num top spans, num top spans] 变为了[num top spans, 50]
        # select top antecedent_mask according to top_antecedent_idx, because some top antecedents are masked.
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx,
                                                device)  # [num top spans, max top antecedents]
        # print('top_antecedent_mask',top_antecedent_mask)
        # print('top_antecedent_mask.shape aft',top_antecedent_mask.shape)

        print('antecedent_offsets.shape bef', antecedent_offsets.shape)
        # 这里同理把antecedent_offsets给截取为[num top spans, 50]
        # select offsets of top antecedent, as distance feature next if needed.
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)
        # 注意这的offsets是bucket前的
        # print('top_antecedent_offsets.shape aft',top_antecedent_offsets.shape)
        # print('top_antecedent_offsets',top_antecedent_offsets)
        ''' The uniform approach of selecting top members is getting the index and the using index to get all 
        corresponding values.
            The uniform approach of pairwise is unsqueeze(vector, 1)+unsqueeze(vector, 0), for values and masks.
        The mask is for illegal pairwises.
        '''

        # Slow mention ranking
        if conf['fine_grained']:
            # print('fine_grained')
            # 4个emb
            # We will use 4 embedding
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                # print('use_metadata') #使用了
                # 取出来top span所述的start的speak(其实取end应该也会一样)
                top_span_speaker_ids = speaker_ids[top_span_starts]  # [num_top_spans]
                # print('top_span_starts.shape', top_span_starts.shape)
                # print('speaker_ids.shape', speaker_ids.shape)
                # print('top_span_speaker_ids.shape', top_span_speaker_ids.shape)

                #  顾名思义，就是top span的50个antecedent所述的speakerid
                """ top_antecedent_idx stores antecedents for each span. The antecedents are the index of span 
                 because antecedents are spans as well.
                """
                top_antecedent_speaker_id = top_span_speaker_ids[
                    top_antecedent_idx]  # [num_top_spans, max_top_antecedents]
                # print('top_antecedent_idx.shape', top_antecedent_idx.shape)
                # print('top_antecedent_speaker_id.shape', top_antecedent_speaker_id.shape)

                # top span的antecedent所属的speakerid是否和这个speakerid一致，它的形状还是[num top spans, 50]
                # Whether a span has same speaker id with its antecedents, True or False. Each row is for a span.
                # [num_top_spans, max_top_antecedents]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                # print('same_speaker.shape', same_speaker.shape)

                # self.emb_same_speaker是Embedding(2, 20)，把True和False变成一个长为20的embedding
                # self.emb_same_speaker is an Embedding(2, 20)
                same_speaker_emb = self.emb_same_speaker(
                    same_speaker.to(torch.long))  # [num top spans, max_top_antecedents, 20]
                # print('self.emb_same_speaker', self.emb_same_speaker)
                # print('same_speaker_emb.shape', same_speaker_emb.shape)
                # 至此，就是把所有的top span和他们的ante是否是一个speak_id取出来，然后做成emb

                # genre is a scalar
                genre_emb = self.emb_genre(genre)  # [20]
                # genre是一个张量
                # genre_emb 是一个长度为20的向量 [20]。到这里就是取出来了 genre的emb,一个长度为20的向量
                # 这个genre应该是属于这个输入的（这个document？），总之这个输入所有的input\sentence都是属于这个genre的
                # print('genre', genre)
                # print('genre_emb.shape', genre_emb.shape)

                # repeat genre_emb, makes it from [20] to [num_top_spans, max_top_antecedents, 20] for convenience
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents,
                                                                                     1)
                # print('genre_emb.shape', genre_emb.shape)
                # 变成一个三维向量，[num_top_spans, max_top_antecedents, 20]，就是变成一个向量对应到所有的[num_top_spans, max_top_antecedents]上
                # 至此，这里的use_metadata是使用了same speaker和genre两个embedding

            if conf['use_segment_distance']:  # 顾名思义，这里就是使用segment距离
                # print('use_segment_distance') #使用了
                # print('input_ids.shape', input_ids.shape)
                # num_segs有几个seg，seg_len分别有多长。这个seg_len还是padding后的，一般是128
                ''' Notice:  segment is different with sentence. A sentence includes some segments.
                   Segment is a word in terms of input. If tokens are input Bert together, they are in 
                   the same segment.
                '''
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1,
                                                                                             seg_len)  # [num_segs, seg_len]
                '''假设num_segs, seg_len = 4, 128，那么token_seg_ids就是
                [[0, 0, ..., 0],
                  [1, 1, ..., 1],
                  [2, 2, ..., 2],
                  [3, 3, ..., 3]
                  ]
                '''
                # print('token_seg_ids', token_seg_ids)
                # print('token_seg_ids.shape', token_seg_ids.shape)
                # 经过这个操作，又会变为1维度。这个操作非常常见在这个代码里。
                # [num_words]
                token_seg_ids = token_seg_ids[input_mask]  # 这里其实就是对于所有的token找出来他们的seg id
                # print('input_mask', input_mask)
                # print('token_seg_ids', token_seg_ids)
                # print('token_seg_ids.shape', token_seg_ids.shape)
                # print('sentence_map.shape', sentence_map.shape)
                # print('token_seg_ids.shape', token_seg_ids)
                # print('sentence_map.shape', sentence_map)

                # seg id of top spans
                top_span_seg_ids = token_seg_ids[top_span_starts]  # 找出top span所属的seg id
                # seg id of top antecedents
                top_antecedent_seg_ids = token_seg_ids[
                    top_span_starts[top_antecedent_idx]]  # 找出top span所拥有的top ante的seg id
                # print('top_antecedent_seg_ids.shape', top_antecedent_seg_ids.shape)
                # top_antecedent_seg_distance是一个二维的。按照我以前的喜好，我喜欢存储成字典。但是我觉得可能这种二维的存储会更方便
                # shape [num_top_spans, max_top_antecedents]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                # 和上面操作类似，这里存储的的是seg id的距离
                print('top_antecedent_seg_distance.shape', top_antecedent_seg_distance.shape)

                # torch.clamp是限制最大最小值。此处是最小为0，最大为self.config['max_training_sentences'] - 1
                # print('top_antecedent_seg_distance bef', top_antecedent_seg_distance)
                # 这里clamp下限0会把负数变成0。效果如下：
                '''top_antecedent_seg_distance bef clamp
                  tensor([[-3, -4, -4,  ..., -4, -4, -3],
                          [ 0, -4, -3,  ..., -4, -4, -3],
                          [ 0,  0, -4,  ..., -4, -4, -3],
                          ...,
                          [ 4,  6,  1,  ...,  5,  6,  4],
                          [ 4,  4,  4,  ...,  4,  4,  0],
                          [ 0,  1,  2,  ...,  3,  1,  2]])
                  top_antecedent_seg_distance aft clamp
                  tensor([[0, 0, 0,  ..., 0, 0, 0],
                          [0, 0, 0,  ..., 0, 0, 0],
                          [0, 0, 0,  ..., 0, 0, 0],
                          ...,
                          [4, 6, 1,  ..., 5, 6, 4],
                          [4, 4, 4,  ..., 4, 4, 0],
                          [0, 1, 2,  ..., 3, 1, 2]])
                '''
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self.config['max_training_sentences'] - 1)
                # print('top_antecedent_seg_distance aft', top_antecedent_seg_distance)

                # 顾名思义，这里也是转换为embedding
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                # print('use_features') #使用了
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                # 这里再次使用了bucket_distance，这次只针对top antecedent，上次是针对所有的在取top antecedent之前
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            # print("conf['coref_depth']",conf['coref_depth']) #在bert设置里，这里coref_depth是1
            for depth in range(conf['coref_depth']):
                ''' 这里的depth的用处是什么？在bert设置里是1，如果是2的话，那么在循环里如果depth是0，就会触发
                    elif depth != conf['coref_depth'] - 1:，从而会触发另外四种high order策略。因为这里没有使用，所以我暂时也没有看
                '''
                # print('depth', depth)
                # print('top_span_emb.shape', top_span_emb.shape)
                # print('top_span_emb', top_span_emb)
                # print('top_antecedent_idx.shape', top_antecedent_idx.shape)
                # print('top_antecedent_idx', top_antecedent_idx)
                ''' top_span_emb is [num top spans, emb size],and top_antecedent_idx is [num top spans, max top antecedents]
                    top_antecedent_idx like： 
                          tensor([[31, 30, 28,  ..., 39, 49, 48],
                                  [ 0, 47, 28,  ..., 39, 48, 23],
                                  [ 1,  0, 47,  ..., 39, 22, 23],
                                  ...,
                                  [ 2,  0, 49,  ..., 47,  9,  3],
                                  [49,  2, 10,  ..., 25, 47, 24],
                                  [50, 49, 23,  ..., 45, 28, 14]]
                    Then we know，top_antecedent_emb stores each top span's antecedent embed
                    (because antecedent is top span as well)
                '''
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                # print('top_antecedent_emb.shape', top_antecedent_emb.shape)
                # print('top_antecedent_emb', top_antecedent_emb)

                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    # print('use_metadata') #使用了
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if conf['use_segment_distance']:
                    # print('use_segment_distance') #使用了
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    # print('use_features') #使用了
                    feature_list.append(top_antecedent_distance_emb)
                # 这个feature_list存储了所有的feature embedding

                feature_emb = torch.cat(feature_list, dim=2)  # 这种操作可以对一个list里的进行concat
                # print('len(feature_list)', len(feature_list))
                # print('feature_list[0].shape', feature_list[0].shape)
                # print('feature_emb.shape', feature_emb.shape)

                feature_emb = self.dropout(feature_emb)
                # print('top_span_emb.shape', top_span_emb.shape)
                # top_span_emb is [Num top, 2324]. after unsqueeze it becomes [Num top, 1, 2324]
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                # print('target_emb.shape', target_emb.shape)
                # repeat(1, max_top_antecedents, 1) 的意思就是第一维和第三维保持不变，第二维重复max_top_antecedents(50)次
                print('top_antecedent_emb.shape', top_antecedent_emb.shape)
                # similarity embedding between each span with their antecedents
                similarity_emb = target_emb * top_antecedent_emb
                # concat all embedding to get a longer embedding
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                # print('pair_emb.shape', pair_emb.shape)

                # Get a score, like converting [169, 50, 7052] to [169, 50, 1]
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                # print('self.coref_score_ffnn(pair_emb).shape', self.coref_score_ffnn(pair_emb).shape)

                # 这里有2个score哎。。。后面还得好好看看
                # print('top_pairwise_slow_scores.shape', top_pairwise_slow_scores.shape)
                # print('top_pairwise_fast_scores.shape', top_pairwise_fast_scores.shape)
                ''' top_pairwise_fast_scores is computed before，is used to select top spans and top antecednets;
                    pairwise_fast_scores = pairwise_mention_score_sum+pairwise_coref_scores+antecedent_distance_score;
                    pairwise_mention_score_sum is the sum of mention_score from span and antecedent;
                    pairwise_coref_scores the dot-produc of embedding of span and its antecedents;

                    top_pairwise_slow_scores is directly computed by FFNN from a huge embedding concated from several embeddings.
                    top_pairwise_slow_scores comes from:
                      target_emb, top_antecedent_emb, similarity_emb, feature_emb. 
                      target_emb is top_span_emb;
                      top_antecedent_emb is top span' top antecedents emb;
                      similarity_emb is element-product of target_emb and top_antecedent_emb;
                      feature_emb includes：
                        same speaker? Yes or not;
                        genre;
                        segment_distance;
                        antecedent_distance;
                '''
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                ''' top_pairwise_fast_scores是之前计算的，是pairwise_fast_scores选取top k个antecednets而来。
                    pairwise_fast_scores是由 pairwise_mention_score_sum,pairwise_coref_scores,antecedent_distance_score
                    构成的。pairwise_mention_score_sum是算好mention_score后把span和antecedent的加起来，pairwise_coref_scores是
                    span和antecedent的embed算内积，antecedent_distance_score是把算好的dis score加上去。

                    而 top_pairwise_slow_scores 是直接从concated后的emb用ffnn算出来的。top_pairwise_slow_scores包括：
                    target_emb, top_antecedent_emb, similarity_emb, feature_emb. 
                    target_emb是top_span_emb，top_antecedent_emb是top span对应的top ante的emb,similarity_emb
                    是target_emb, top_antecedent_emb点乘的emb,feature_emb是：
                    same speaker? Yes or not;
                    genre;
                    segment_distance;
                    antecedent_distance;
                '''
                # print('top_pairwise_scores.shape', top_pairwise_scores.shape)

                # print("conf['higher_order']", conf['higher_order']) # attended_antecedent
                # print("conf['coref_depth']", conf['coref_depth'])
                # print('depth', depth)
                if conf['higher_order'] == 'cluster_merging':
                    # print("conf['higher_order']", conf['higher_order']) #这里没有使用
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                                self.emb_cluster_size, self.cluster_score_ffnn, None,
                                                                self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'],
                                                                easy_cluster_first=conf['easy_cluster_first'])
                    break

                elif depth != conf['coref_depth'] - 1:
                    # print("conf['higher_order']", conf['higher_order']) # 这里也没有使用
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                                  device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                             device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx,
                                                                  top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                              self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        # print("do_loss", do_loss)
        if not do_loss:
            # print("do_loss") # do_loss为T，这里不走
            if conf['fine_grained'] and conf['higher_order'] == 'cluster_merging':
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)  # [num top spans, max top antecedents + 1]
            return candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores

        # Get gold labels
        # print('top_span_cluster_ids.shape 1', top_span_cluster_ids.shape)
        # print('top_antecedent_idx.shape 2', top_antecedent_idx.shape)
        ''' 这里top_span_cluster_ids是一维的，top_antecedent_idx是二维的。以[156]、[156, 50]举例，
        top_span_cluster_ids[top_antecedent_idx] 是对里面的[156, 50]个scalar分别在[156]里寻找，然后再保持[156, 50]
        的形状。
        这种写法真的好简洁，但是没有那么的易懂。不知道有没有其他好的写法？
        '''

        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        # print('top_antecedent_cluster_ids.shape 3', top_antecedent_cluster_ids.shape)
        # For False (0) in top_antecedent_mask, will -100000. For True (1), will -0.
        top_antecedent_cluster_ids += (top_antecedent_mask.to(
            torch.long) - 1) * 100000  # Mask id on invalid antecedents
        # 减1的话就是把1和0变成了0和-1，乘100000就是把-1放大了
        # print('top_antecedent_mask.to(torch.long)- 1', top_antecedent_mask.to(torch.long)- 1)
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids,
                                                                                     1))  # [num top spans, max top antecedents]
        '''top_span_cluster_ids是一维的，以222为例，unsqueeze把它变成[222,1].top_antecedent_cluster_ids是二维的，是[222,50]
        same_gold_cluster_indicator就是判断所有的top span是不是和它所对应的top antecedent是一致的cluster
        '''
        # print('top_antecedent_cluster_ids.shape', top_antecedent_cluster_ids.shape)
        # print('top_span_cluster_ids.shape', top_span_cluster_ids.shape)

        # whether it is a mention
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)  # [top span num, 1]
        # 有cluster_id，意味着它不是一个dummy_indicator
        # non_dummy_indicator判断是不是有antecedent
        # print('non_dummy_indicator', non_dummy_indicator)
        # print('non_dummy_indicator.shape', non_dummy_indicator.shape)

        # (Span has same cluster id with an candidate) & (span is a mention). [top span num, max top antecedents]
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        # pairwise_labels里如果为True，则这些ancetedent和top span是同一个cluster且这个top span不是一个dummy_indicator（cluster id>0)
        # 这些top span和他们的ante是不是pair，维度是[top span num, 50]，取值是True或者False。他的意思就是是否配对的label
        # print('pairwise_labels', pairwise_labels)
        # print('pairwise_labels.shape', pairwise_labels.shape)
        # print('pairwise_labels.sum(axis=0)', pairwise_labels.sum(axis=1)) #有的top span，所有的pairwise_labels都是0！
        # print('pairwise_labels.sum(axis=0).shape', pairwise_labels.sum(axis=1).shape)

        ''' pairwise_labels.any(dim=1, keepdims=True) means in dim=1, if any value is 1, return [1].
        logical_not means if 1 return False, and if 0 return True.
        dummy_antecedent_labels means whether each span has no referred candidate.  
        '''
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))  # [top span num, 1]
        # 在dim=1这个维度上，任意一个值为1.就是在[维度0，维度1]的维度1上有任何一个为True。经过这个操作后，会从[维度0，1]
        # torch.logical_not是把逻辑反过来，原来是True的变为False，反之亦然
        # 这里的意思是说没有任何一个配对的，则把当前这个span变为一个dummy
        # print('pairwise_labels',pairwise_labels)
        # print('pairwise_labels.shape',pairwise_labels.shape)
        # print('pairwise_labels.any(dim=1, keepdims=True)', pairwise_labels.any(dim=1, keepdims=True))
        # print('pairwise_labels.any(dim=1, keepdims=True).shape', pairwise_labels.any(dim=1, keepdims=True).shape)
        # print('dummy_antecedent_labels',dummy_antecedent_labels)
        # print('dummy_antecedent_labels.shape',dummy_antecedent_labels.shape)

        # The first column is whether this span has no referred cadidate. The next columns is whether referred with candidates
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)
        # print('top_antecedent_gold_labels.shape', top_antecedent_gold_labels.shape) #现在是[top span num, 51]
        # 51的后50个表示是否50个antecedent是和当前的top span一个label，而第0表示是否它是一个dummy span(在当前span的clusterid=0或者没有任何一个clusterid与他相同)
        # print('top_antecedent_gold_labels', top_antecedent_gold_labels)

        # Get loss
        # the first column is 0
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        # torch.zeros(num_top_spans, 1, device=device)是[top span num, 1]。于是top_antecedent_scores是[top span num, 51]
        # print('torch.zeros(num_top_spans, 1, device=device).shape', torch.zeros(num_top_spans, 1, device=device).shape)
        # print('top_pairwise_scores.shape', top_pairwise_scores.shape)
        # print('top_antecedent_scores', top_antecedent_scores)
        # print('top_antecedent_scores.shape', top_antecedent_scores.shape)
        # 简而言之，这个操作把top_pairwise_scores [top span num, 50]的最左边加上一个0，变成[top span num, 51]
        # 目前已经有了配对分数，也有了配对标识

        # print("conf['loss_type']", conf['loss_type']) # marginalized
        if conf['loss_type'] == 'marginalized':
            '''logsumexp： https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/
              logsumexp returns a number slightly larger then the max value. It is similar with Max but it is differentiable.
              torch.log(1)=0, torch.log(0)=-inf. Its role is masking. When top_antecedent_gold_labels is False, 
              top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)) = -inf,
              exp(-inf)=0, so it brings no influence on logsumexp. Only when the top_antecedent_gold_labels is True, the 
              top_antecedent_scores will be computed in logsumexp.
            '''
            log_marginalized_antecedent_scores = torch.logsumexp(top_antecedent_scores + \
                                                                 torch.log(top_antecedent_gold_labels.to(torch.float)), \
                                                                 dim=1)
            # logsumexp： https://pytorch.org/docs/stable/generated/torch.logsumexp.html
            # 对配对label top_antecedent_gold_labels进行Log,如果是true，则变为0，如果为False,则为-inf。为-inf则会mask掉top_antecedent_scores
            # 里的值，为0则会放行里面的值
            # 如果有配对关系，则取对应的top_antecedent_scores值，如果没有配对关系，则变为-inf，有配对关系的趋向于0.相当于一个mask的作用
            # print('top_antecedent_gold_labels.to(torch.float)', top_antecedent_gold_labels.to(torch.float))
            # print('torch.log(top_antecedent_gold_labels.to(torch.float))', torch.log(top_antecedent_gold_labels.to(torch.float)))
            # print('top_antecedent_scores', top_antecedent_scores)
            '''log_marginalized_antecedent_scores 里会有很多0，那是因为如下这种造成的：
            [0., -inf, -inf, ... -inf] 只有第一个值是0，代表着top_antecedent_gold_labels也是[0., -inf, -inf, ... -inf]
            且top_antecedent_scores为[0.0, x, x, ..., x]。只有第一个被放行了，代表着它和任何antecedent没有coference关系。
            '''
            # # 测试：
            # sum = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            # print('sum', top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)))
            # print('log_marginalized_antecedent_scores', log_marginalized_antecedent_scores)
            # if log_marginalized_antecedent_scores.any():
            #     idx = log_marginalized_antecedent_scores.argsort(descending=True)
            #     max_idx = idx[0]
            #     min_idx = idx[-1]
            #     print('max_idx', max_idx)
            #     print('min_idx', min_idx)
            #     print('sum[max_idx]', sum[max_idx])
            #     print('sum[min_idx]', sum[min_idx])
            #     print('top_antecedent_gold_labels max', torch.log(top_antecedent_gold_labels.to(torch.float))[max_idx])
            #     print('top_antecedent_gold_labels min', torch.log(top_antecedent_gold_labels.to(torch.float))[min_idx])
            #     print('top_antecedent_scores[max_idx]', top_antecedent_scores[max_idx])
            #     print('top_antecedent_scores[min_idx]', top_antecedent_scores[min_idx])
            #     log_norm_test = torch.logsumexp(top_antecedent_scores, dim=1)
            #     # print('log_norm_test', log_norm_test)
            #     print('log_norm_test[max_idx]', log_norm_test[max_idx])
            #     print('log_norm_test[min_idx]', log_norm_test[min_idx])

            '''因为logsumexp会取一个比最大值稍微大一点的值，对于没有任何coference的span (log_marginalized_antecedent_scores对应值为0)，
            它在top_antecedent_scores会是
            [0.0, x, x, ..., x]，而log_marginalized_antecedent_scores是0.0（如上所述）。而top_antecedent_scores里的最大值会训练的
            让他小于0，从而log_norm是一个比0略大或者小的值，从而最小化loss。
            如下是具体实例：
            sum[max_idx] tensor([0., -inf, -inf, ..., -inf] )
            top_antecedent_gold_labels max tensor([0., -inf,  , -inf]
            top_antecedent_scores[max_idx] tensor([ 0.0000e+00,  1.9845e+01,  1.0602e+01])

            对于有coference的，如下：以idx=37为例
            min_idx = 37
            log_marginalized_antecedent_scores tensor([  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000, -53.5065,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000])
            第37个值不为0，为-53.5065，它的sum只如下：
            sum[min_idx] tensor([    -inf,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
              -53.5065,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf])
            拆解，可以看到它的37位是和span有共同的cluster，有-53.5065这个值
            top_antecedent_gold_labels min tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,
            -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,
            -inf, -inf, -inf])       
            top_antecedent_scores[min_idx] tensor([  0.0000,   5.5987,  -4.5748,  -5.7887,  -6.3555,  -8.2518, -10.7151,
              -13.6023, -10.8690, -11.7486, -14.5348, -16.4221, -16.7845, -17.0610,
              -19.8137, -20.2194, -22.2871, -24.5872, -22.7082, -26.4905, -29.6787,
              -28.9808, -34.6860, -34.8431, -37.2729, -36.8012, -39.3922, -41.6048,
              -41.1821, -43.5197, -44.0913, -48.3450, -52.6722, -50.9655, -53.6990,
              -53.5065！！, -57.3499, -60.2096,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf,     -inf,     -inf,     -inf,     -inf,     -inf,
                  -inf,     -inf])
            而对于log_norm, log_norm_test[min_idx] tensor(5.6024, grad_fn=<SelectBackward0>)
            它算出来的值是比最大值5.5987略大的5.6024。
            如果要降低loss，就需要让37位的值是最大的，从而比它略大的值成为log_norm。

            也就是说，如果有coreference，则会训练让对应coference的top_antecedent_scores最大，如果没有coreference，
            则会让第一个值（0.0）最大，从而让这个span成为dummy_antecedent.
            '''

            # print('log_marginalized_antecedent_scores.shape', log_marginalized_antecedent_scores.shape)
            # 因为在dim=1维度上进行的logsumexp，所以相当于对所有的top span，计算pairlabel为1的分数。故而
            # log_marginalized_antecedent_scores的维度是[top span num]一个一维的向量

            # use logsumexp without masking
            log_norm = torch.logsumexp(top_antecedent_scores, dim=1)  # 取log后的score，[top span num]一个一维的向量
            # print('top_antecedent_scores', top_antecedent_scores)
            # print('log_norm', log_norm)
            # print('log_norm.shape', log_norm.shape)

            ''' How can loos be minimized? 
            For span with no antecedent, 
              top_antecedent_gold_labels.to(torch.float) = [1,0,0,...,0]
              log(top_antecedent_gold_labels.to(torch.float)) = [0,-inf,-inf,...,-inf]
              top_antecedent_scores = [0.0, 3.21, 4.01,...,6.89]
              log_marginalized_antecedent_scores = log(exp(0)+exp(-inf)+...+exp(-inf))=log(1)=0
              if we want log_norm - log_marginalized_antecedent_scores be minimized, 
              we know logsumexp(top_antecedent_scores) ~= max(top_antecedent_scores), we should let 0 is the max value 
              in top_antecedent_scores. We know 0 is the first value in top_antecedent_scores.

            For span with antecedents,
              top_antecedent_gold_labels.to(torch.float) = [0,0,0,.,1,.,0]
              log(top_antecedent_gold_labels.to(torch.float)) = [-inf,-inf,-inf,.,0,.,-inf]
              top_antecedent_scores = [0.0, 3.21, 4.01,.,8.99,.,6.89]
              log_marginalized_antecedent_scores = log(exp(-inf)+exp(-inf)+.,exp(8.99),.+exp(-inf))=log(8.99)
              if we want log_norm - log_marginalized_antecedent_scores be minimized, 
              we know logsumexp(top_antecedent_scores) ~= max(top_antecedent_scores), we should let 8.99 is the max value 
              in top_antecedent_scores.  

            '''
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
            '''log_norm是相似分数，而log_marginalized_antecedent_scores是把label为0(表示无关联）mask掉(设置为0)后
               让log_norm - log_marginalized_antecedent_scores越接近，就是让mask掉的分数趋向于-inf，而没有mask掉的（有配对关系的）趋近于0
               只有这样loss才能降低
            '''
            # torch.log(top_antecedent_gold_labels.to(torch.float))越接近0，注意这里很多都是-inf，所以加了top_antecedent_scores
            # 之后就会变为-inf，起到一个mask的作用
            # print('torch.log(top_antecedent_gold_labels.to(torch.float)', torch.log(top_antecedent_gold_labels.to(torch.float)))
            '''
            下面的都没有使用，因此就到这里为之了。可以知道的是，如果是正确的antecedent被选出来，那么他和top span的配对分数
            会训练让他们接近于0，而和top span不配对的那些配对分数会让他们趋向于-inf。如果多次训练，就会越来越有更多对的top span
            被选出来，并且他们的top antecedent也会越来越正确。
            '''

        elif conf['loss_type'] == 'hinge':  # 这里没有用到，先不看了
            top_antecedent_mask = torch.cat(
                [torch.ones(num_top_spans, 1, dtype=torch.bool, device=device), top_antecedent_mask], dim=1)
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedent_scores, dim=1)
            gold_antecedent_scores = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - conf['false_new_delta']) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - conf['false_new_delta']) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # Add mention loss
        # print("conf['mention_loss_coef']", conf['mention_loss_coef']) #0
        if conf['mention_loss_coef']:
            # print("conf['mention_loss_coef']", conf['mention_loss_coef']) # 这里没用到，先不看了
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores))) * conf[
                'mention_loss_coef']
            loss += loss_mention

        # print("conf['higher_order']", conf['higher_order'])
        if conf['higher_order'] == 'cluster_merging':
            # print("conf['higher_order']", conf['higher_order']) # 这里没用到，先不看了
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if conf['cluster_dloss']:
                loss += loss_cm
            else:
                loss = loss_cm

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (
                num_top_spans, (top_span_cluster_ids > 0).sum(), (top_span_cluster_ids > 0).sum() / num_top_spans))
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention)
                if conf['loss_type'] == 'marginalized':
                    logger.info(
                        'norm/gold: %.4f/%.4f' % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores)))
                else:
                    logger.info('loss: %.4f' % loss)
        self.update_steps += 1

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedent_idx, top_antecedent_scores], loss

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends,
                                                                                   antecedent_idx, antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

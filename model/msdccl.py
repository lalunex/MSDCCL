from utils.utils import get_model
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender
from torch.nn.init import xavier_uniform_, xavier_normal_
from functools import reduce


class MSDCCL(SequentialRecommender):

    def __init__(self, config, dataset):
        super(MSDCCL, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.embedding_size = config['embedding_size']
        self.our_ae_drop_out = config['our_ae_drop_out']
        self.our_att_drop_out = config['our_att_drop_out']

        # 下面是我添加的参数
        self.sequence_last_m = config['sequence_last_m']
        self.transformer_encoder_heads = config['transformer_encoder_heads']
        self.transformer_encoder_layers = config['transformer_encoder_layers']
        self.transformer_encoder_dim_feedforward = config['transformer_encoder_dim_feedforward']
        self.transformer_encoder_layer_norm_eps = config['transformer_encoder_layer_norm_eps']
        self.reweight_loss_alpha = config['reweight_loss_alpha']
        self.reweight_loss_lambda = config['reweight_loss_lambda']
        self.train_batch_size = config['train_batch_size']
        self.reweight_loss_theta_before = 0  # 初始化
        self.reweight_loss_theta_after = 0  # 初始化
        self.gumbel_tau = 0.5  # 初始化
        self.spu_cl_tau = 0.5  # 初始化
        self.train_set_ratio = config['train_set_ratio']

        self.n_users = dataset.user_num

        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.emb_dropout = nn.Dropout(self.our_ae_drop_out)
        self.relu = nn.ReLU()

        # 下面是添加的caser代码来读取短期信息，初始化模型
        self.user_short_nh = config['user_short_nh']
        self.user_short_nv = config['user_short_nv']
        # vertical conv layer
        self.user_short_conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.user_short_nv,
            kernel_size=(self.sequence_last_m + 1, 1))
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.sequence_last_m + 1)]
        self.user_short_conv_h = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.user_short_nh,
                      kernel_size=(i, self.hidden_size)) for i in lengths
        ])
        # fully-connected layer
        self.user_short_fc_dim_v = self.user_short_nv * self.hidden_size
        self.user_short_fc_dim_h = self.user_short_nh * len(lengths)
        user_short_fc_dim_in = self.user_short_fc_dim_v + self.user_short_fc_dim_h
        self.user_short_fc = nn.Linear(user_short_fc_dim_in, self.hidden_size)

        # 下面是添加的transformerencoder代码来读取长期信息，初始化模型
        self.user_long_transformer_encoder = TransformerEncoder(
            n_layers=self.transformer_encoder_layers,
            n_heads=self.transformer_encoder_heads,
            hidden_size=self.hidden_size,
            inner_size=self.transformer_encoder_dim_feedforward,
            hidden_dropout_prob=self.our_ae_drop_out,
            attn_dropout_prob=self.our_att_drop_out,
            hidden_act='relu',
            layer_norm_eps=self.transformer_encoder_layer_norm_eps
        )

        # 下面是添加的transformerencoder代码来聚集hard信息，初始化模型
        self.hard_info_transformer_encoder = TransformerEncoder(
            n_layers=self.transformer_encoder_layers,
            n_heads=self.transformer_encoder_heads,
            hidden_size=self.hidden_size,
            inner_size=self.transformer_encoder_dim_feedforward,
            hidden_dropout_prob=self.our_ae_drop_out,
            attn_dropout_prob=self.our_att_drop_out,
            hidden_act='relu',
            layer_norm_eps=self.transformer_encoder_layer_norm_eps
        )

        # 下面是将用户长期和短期兴趣融合起来的MLP
        self.user_short_or_long_map = nn.Linear(
            self.embedding_size, self.embedding_size, False)
        self.user_short_and_long_fusion = nn.Linear(
            self.embedding_size, self.embedding_size, True)

        # 下面是将用户兴趣与原来序列做一个attention的初始化
        self.attention_read_out = AttnReadout(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=2,
            layer_norm=True,
            feat_drop=self.our_att_drop_out,
        )

        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)
        self.binary_softmax = nn.Softmax(dim=-1)

        self.recommender_ce_curriculum_loss = nn.CrossEntropyLoss(reduction='none')

        self.apply(self._init_weights)

        # 初始化sub_model
        self.sub_model = get_model(config['sub_model'])(config, dataset).to(config['device'])
        self.sub_model_name = config['sub_model']
        self.item_embedding = self.sub_model.item_embedding

        if config['load_pre_train_emb'] is not None and config['load_pre_train_emb']:
            checkpoint_file = config['pre_train_model_dict'][config['dataset']][config['sub_model']]
            checkpoint = torch.load(checkpoint_file)
            if config['sub_model'] == 'DSAN':
                embedding_weight = checkpoint['state_dict']['embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                item_embedding_weight = checkpoint['state_dict']['item_embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(item_embedding_weight, freeze=False)

    # 在apply里面使用的初始化函数
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    # 下面是使用最后一个item做短期信息，forward部分
    def user_short_interest_extract(self, item_seq, item_seq_len, target_item=None):
        # 下面是取序列最后面的m个item
        sequence_last_m_interval_end = torch.squeeze(item_seq_len)
        sequence_last_m_interval_start = sequence_last_m_interval_end - self.sequence_last_m
        sequence_last_m_indexes = []
        for i in range(item_seq_len.shape[0]):
            if sequence_last_m_interval_start[i] >= 0:
                item_index = item_seq[i, sequence_last_m_interval_start[i]:sequence_last_m_interval_end[i]]
                sequence_last_m_indexes.append(item_index)
            else:
                item_index = item_seq[i, 0:sequence_last_m_interval_end[i]]
                item_index = torch.nn.ZeroPad2d(padding=(self.sequence_last_m - sequence_last_m_interval_end[i], 0))(
                    item_index)
                sequence_last_m_indexes.append(item_index)
        sequence_last_m_indexes = torch.stack(sequence_last_m_indexes, dim=0)
        user_short_items_indexes = torch.nn.ZeroPad2d(padding=(1, 0, 0, 0))(
            sequence_last_m_indexes)
        if target_item is not None:
            user_short_items_indexes = torch.cat([sequence_last_m_indexes, target_item], dim=-1)

        user_short_items_emb_ori = self.item_embedding(user_short_items_indexes)

        # calculate the mask
        mask = torch.ones(
            user_short_items_indexes.shape, dtype=torch.float,
            device=item_seq.device) * user_short_items_indexes.gt(0)

        # size = [batchSize, user_num_per_batch, 1]
        mask = mask.unsqueeze(2)
        user_short_items_emb = self.emb_dropout(user_short_items_emb_ori) * mask
        user_short_items_emb = user_short_items_emb.unsqueeze(1)

        # Convolutional Layers，下面的语句是文中的卷积代码
        out, out_h, out_v = None, None, None

        # vertical conv layer
        if self.user_short_nv:
            out_v = self.user_short_conv_v(user_short_items_emb)
            out_v = out_v.view(-1, self.user_short_fc_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.user_short_nh:
            for conv in self.user_short_conv_h:
                conv_out = self.relu(conv(user_short_items_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.emb_dropout(out)
        # fully-connected layer
        user_short_output = self.relu(self.user_short_fc(out))

        return user_short_output

    # 下面是添加的transformerencoder代码来读取长期信息，forward部分
    def sequence_info_extract(self, item_seq, item_seq_len, mask, mode='user_long'):
        item_seq = item_seq.long()
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        items_index = item_seq
        items_emb_ori = self.item_embedding(items_index)
        items_emb = items_emb_ori + position_embedding
        items_emb = self.LayerNorm(items_emb)
        items_emb = self.emb_dropout(items_emb) * mask
        extended_attention_mask = self.get_attention_mask(item_seq)

        # 下面的语句是transformerencoder的代码
        if mode == 'user_long':
            transformer_encoder = self.user_long_transformer_encoder
        elif mode == 'hard_info':
            transformer_encoder = self.hard_info_transformer_encoder
        else:
            raise ValueError('there is no mode for the transformers module.')
        outputs = transformer_encoder(
            items_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    # 下面是添加的MLP代码来融合短期长期信息，forward部分
    def user_interest_fusion(self, user_short_output, user_long_output):
        user_short_output_mapping = self.user_short_or_long_map(user_short_output)
        user_long_output_mapping = self.user_short_or_long_map(user_long_output)
        user_interest_fusion_emb = self.user_short_and_long_fusion(
            self.relu(user_short_output_mapping + user_long_output_mapping)
        )
        return user_interest_fusion_emb

    # 下面是使用任意的下游模型来做recommender以得到序列表征，forward部分
    def sub_model_forward(self, item_seq, item_seq_emb, item_seq_len, user):
        if self.sub_model_name == 'BERT4Rec':
            sub_model_items_output = self.sub_model.forward(item_seq, item_seq_emb)
        elif self.sub_model_name == 'GRU4Rec':
            sub_model_items_output = self.sub_model.forward(item_seq_emb, item_seq_len)
        elif self.sub_model_name == 'SASRec':
            sub_model_items_output = self.sub_model.forward(item_seq, item_seq_emb, item_seq_len)
        elif self.sub_model_name == 'Caser':
            sub_model_items_output = self.sub_model.forward(user, item_seq_emb)
        elif self.sub_model_name == 'NARM':
            sub_model_items_output = self.sub_model.forward(item_seq, item_seq_emb, item_seq_len)
        elif self.sub_model_name == 'STAMP':
            sub_model_items_output = self.sub_model.forward(item_seq_emb, item_seq_len)
        else:
            raise ValueError(f'Sub_model [{self.sub_model_name}] not support.')
        return sub_model_items_output

    r"""该函数是为了将降噪的item序列里面的最后一个拿出来
    generated_seq：经降噪的模型
    seq_output：子模型（如bertrec等）的输出
    """

    def seq_last_one_emb_extract(self, sub_model_items_output, item_seq_len):
        if self.sub_model_name in ['Caser', 'GRU4Rec', 'NARM', 'DSAN', 'STAMP']:
            sub_model_seq_output = sub_model_items_output
        else:
            sub_model_seq_output = self.gather_indexes(sub_model_items_output, item_seq_len - 1)
        return sub_model_seq_output  # [B H]

    # 下面是使用了gumbel-softmax来对序列进行分类以获得噪音或非噪音item，forward部分
    def get_pos_and_neg_item(self, item_seq_len, score, mask):
        mask = mask.squeeze()

        score_gumbel_softmax = F.gumbel_softmax(score, tau=self.gumbel_tau, hard=True)
        neg_flag = score_gumbel_softmax[:, :, 1] * mask
        pos_flag = (1 - neg_flag) * mask

        r"""
        下面的代码是为了获得position items
        """
        pos_seq_len = torch.sum(pos_flag, dim=-1)
        # 以防止某个序列内没有正样本，而导致的supervised contrastive learning的分母为0
        pos_flag[pos_seq_len.eq(0), 0] = 1

        clean_seq_percent = torch.sum(pos_seq_len, dim=0) / item_seq_len.sum() * 100

        return pos_flag, clean_seq_percent

    # 下面是使用hard信息来获取序列的emb，forward部分
    def hard_info_seq_emb(self, item_seq, pos_flag, mask):
        pos_seq = item_seq * pos_flag
        pos_seq_len = torch.sum(pos_flag, dim=-1)
        pos_seq_len = pos_seq_len.type(torch.long).to(self.device)
        row_indexes, col_id = torch.where(pos_seq.gt(0))
        hard_denoising_seq = torch.zeros_like(pos_seq)
        pos_id_list = [torch.arange(i).tolist() for i in pos_seq_len]
        pos_id_list_concat = reduce((lambda x, y: x + y), pos_id_list)
        pos_id_concat = torch.tensor(pos_id_list_concat, device=self.device)
        hard_denoising_seq[row_indexes, pos_id_concat] = pos_seq[row_indexes, col_id]

        return hard_denoising_seq, pos_seq_len

    # 下面是使用target的信息来监督hard的信息的生成
    def cross_entropy_loss(self, interaction, seq_emb, islist=True):
        all_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token

        # 下面开始将e与每个item进行交互（点乘），并计算cross entropy loss
        all_items_scores = torch.matmul(seq_emb, all_items_emb.transpose(0, 1))  # [B, item_num]
        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        target_item = target_item.squeeze()
        recommender_ce_loss_list = self.recommender_ce_curriculum_loss(all_items_scores, target_item)

        return recommender_ce_loss_list if islist else recommender_ce_loss_list.mean()

    def bpr_loss_list(self, seq_emb, target_item, neg_item):
        target_item_emb = self.item_embedding(target_item.squeeze()).unsqueeze(-1)
        neg_item_emb = self.item_embedding(neg_item).unsqueeze(-1)
        seq_emb = seq_emb.unsqueeze(-2)

        pos_score = torch.matmul(seq_emb, target_item_emb)
        neg_score = torch.matmul(seq_emb, neg_item_emb)

        losses = -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).squeeze()
        return losses

    # 下面是使用了supervised contrastive learning来对分类好的item计算loss，forward部分（batch mean）
    def supervised_contrastive_learning(
            self,
            seq_emb,
            item_seq_emb,
            item_pos_flag,
            mask,
            tau):
        item_seq_emb = item_seq_emb * mask

        target_seq_emb = seq_emb.unsqueeze(1).expand_as(
            item_seq_emb
        ).to(torch.float)
        # 这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(item_seq_emb, target_seq_emb, dim=2)

        # 这步给相似度矩阵求exp, 并且除以温度参数T, 注意要乘mask
        similarity_matrix_after_exp = torch.exp(similarity_matrix / tau) * mask.squeeze()

        # 这步产生了正样本（五噪音item）的相似度矩阵，其他位置都是0
        sim = item_pos_flag * similarity_matrix_after_exp

        # 用原先的相似度矩阵减去正样本矩阵得到负样本（噪音item）的相似度矩阵
        no_sim = similarity_matrix_after_exp - sim
        # 把负样本矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim, dim=1, keepdim=True)

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是正样本矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个正样本的相似度，就是分子的数据。
        '''
        no_sim_sum_expend = no_sim_sum.expand_as(item_pos_flag)
        sim_sum = sim + no_sim_sum_expend

        # 为了防止自监督对比学习的分母为0
        zero_anomaly_process = sim_sum.le(0.).float() * 1e-10
        anomaly_process_sim_sum = sim_sum + zero_anomaly_process

        sim_div = torch.div(sim, anomaly_process_sim_sum)

        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        sim_div_sum = sim_div + sim_div.eq(0)

        # 接下来就是算一个批次中的sup_con_loss了，batch里面per item的loss
        sim_div_sum_log = -torch.log(sim_div_sum)  # 求-log
        sup_con_losses = torch.sum(sim_div_sum_log, dim=1) / (torch.sum(item_pos_flag, dim=-1))

        return sup_con_losses

    def forward(self, interaction, train_flag=True):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN].unsqueeze(1)
        user = interaction[self.USER_ID]

        target_item = None
        if train_flag:
            target_item = interaction[self.ITEM_ID].unsqueeze(1)

        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)

        # size = [batchSize, user_num_per_batch, 1]
        mask = mask.unsqueeze(2)

        # 下面是forward里面的短期兴趣提取器
        user_short_output = self.user_short_interest_extract(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            target_item=target_item
        )

        # 下面是forward里面的长期兴趣提取器
        user_long_output = self.sequence_info_extract(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            mask=mask,
            mode='user_long'
        )

        # 下面是forward里面的短期长期兴趣融合器
        user_interest_fusion_emb = self.user_interest_fusion(
            user_short_output=user_short_output, user_long_output=user_long_output)

        # 下面是添加的attention代码来获取降噪后的序列embedding和2为的注意力分数，forward部分
        # socre: [batchsize, max_seq_len, 2]; denoised_items_emb: [batchsize, max_seq_len, hidden_dim]
        item_seq_emb = self.item_embedding(item_seq)
        score, denoised_items_emb = self.attention_read_out(item_seq_emb, user_interest_fusion_emb, mask)

        # 下面是使用任意的下游模型来做recommender以得到序列表征，forward部分
        soft_info_items_output = self.sub_model_forward(
            item_seq=item_seq,
            item_seq_emb=denoised_items_emb,
            item_seq_len=item_seq_len,
            user=user
        )

        # 下面是取出序列捏最后一个item的embedding作为序列表征，forward部分
        # sub_model_seq_output: [B, H]
        soft_info_seq_emb = self.seq_last_one_emb_extract(
            sub_model_items_output=soft_info_items_output,
            item_seq_len=item_seq_len)

        # 下面是使用了gumbel-softmax来对序列进行分类以获得噪音或非噪音item，forward部分
        reorder_pos_flag, clean_seq_percent = self.get_pos_and_neg_item(
            item_seq_len=item_seq_len,
            score=score,
            mask=mask
        )

        return soft_info_seq_emb, reorder_pos_flag, clean_seq_percent, user_interest_fusion_emb, soft_info_seq_emb

    def calculate_reweight_loss(self, interaction, drop_rate, gumbel_tau, spu_cl_tau):
        self.gumbel_tau = gumbel_tau
        self.spu_cl_tau = spu_cl_tau
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_emb = self.item_embedding(item_seq)
        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        neg_item = interaction[self.NEG_ITEM_ID]

        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)
        mask = mask.unsqueeze(2)

        # foward是为了得到e，sup_con_loss和sequence的干净程度（含有非噪音item的程度）
        sub_model_seq_output, reorder_pos_flag, clean_seq_percent, _, _ = self.forward(interaction)

        # 下面是使用了transformer来对去噪的序列进行编码的部分，forward
        hard_denoising_seq, pos_seq_len = self.hard_info_seq_emb(
            item_seq=item_seq,
            pos_flag=reorder_pos_flag,
            mask=mask
        )
        pos_mask = torch.ones(hard_denoising_seq.shape, dtype=torch.float,
                              device=hard_denoising_seq.device) * hard_denoising_seq.gt(0)
        pos_mask = pos_mask.unsqueeze(2)
        hard_info_seq_emb = self.sequence_info_extract(hard_denoising_seq, pos_seq_len, pos_mask, 'hard_info')
        hard_bpr_losses = self.bpr_loss_list(hard_info_seq_emb, target_item, neg_item)

        # 关于hard的信息的loss
        # hard_recommender_ce_loss = self.cross_entropy_loss(interaction, hard_info_seq_emb, False)

        # 下面是使用了supervised contrastive learning（batch mean）对分类好的item计算loss，forward部分
        sup_con_losses = self.supervised_contrastive_learning(
            seq_emb=sub_model_seq_output,
            item_seq_emb=item_seq_emb,
            item_pos_flag=reorder_pos_flag,
            mask=mask,
            tau=self.spu_cl_tau)

        soft_recommender_ce_loss_list = self.cross_entropy_loss(interaction, sub_model_seq_output, True)
        recommender_ce_loss_seq_length = len(soft_recommender_ce_loss_list)


        # 下面是做不同训练集大小的实验结果的代码
        mask_weights = torch.ones_like(hard_bpr_losses) # create a Tensor of weights
        mask_weights = torch.multinomial(mask_weights, int(hard_bpr_losses.shape[0] * self.train_set_ratio))
        final_ratio_mask = torch.zeros_like(hard_bpr_losses) # create a Tensor of weights
        final_ratio_mask[mask_weights] = 1

        hard_bpr_losses = hard_bpr_losses * final_ratio_mask
        sup_con_losses = sup_con_losses * final_ratio_mask
        soft_recommender_ce_loss_list = soft_recommender_ce_loss_list * final_ratio_mask

        # 求的batch里面的instance层次的loss
        hard_bpr_loss = torch.sum(hard_bpr_losses) / final_ratio_mask.count_nonzero()
        sup_con_loss = torch.sum(sup_con_losses) / final_ratio_mask.count_nonzero()
        recommender_ce_loss_seq_length = final_ratio_mask.count_nonzero()

        soft_recommender_ce_loss = self.calculate_curriculum_loss(
            loss_list=soft_recommender_ce_loss_list,
            drop_rate=drop_rate,
            seq_length=recommender_ce_loss_seq_length)
        # # 求的batch里面的instance层次的loss
        # hard_bpr_loss = hard_bpr_losses.mean()
        # sup_con_loss = torch.sum(sup_con_losses) / item_seq_emb.shape[0]

        # # 这里是recommender推荐item的地方，这里用到了课程学习——从易到难，返回计算好的课程学习loss（batch mean）
        # soft_recommender_ce_loss = self.calculate_curriculum_loss(
        #     loss_list=soft_recommender_ce_loss_list,
        #     drop_rate=drop_rate,
        #     seq_length=recommender_ce_loss_seq_length)
        
        # 下面的代码是为了检测课程学习的可用性
        # recommender_ce_loss = torch.sum(recommender_ce_loss_list)

        # 下面是reweight loss function
        # soft_recommender_ce_loss = soft_recommender_ce_loss_list.mean()
        # total_loss = self.reweight_loss(
        #     sup_con_curriculum_loss=sup_con_loss,
        #     recommender_ce_loss=recommender_ce_loss)
        # total_loss = soft_recommender_ce_loss + hard_bpr_loss * self.reweight_loss_lambda + sup_con_loss
        total_loss = soft_recommender_ce_loss + (hard_bpr_loss + sup_con_loss) * self.reweight_loss_lambda

        # 下面的代码是为了检测对比学习的可靠性
        # recommender_ce_loss_div = torch.div(
        #     recommender_ce_loss,
        #     torch.tensor(float(self.train_batch_size), requires_grad=True))
        # return recommender_ce_loss_div, clean_seq_percent

        return total_loss, clean_seq_percent

    def reweight_loss(self, sup_con_curriculum_loss, recommender_ce_loss):
        # sup_con_curriculum_loss_div = torch.div(
        #     sup_con_curriculum_loss,
        #     torch.tensor(float(self.train_batch_size), requires_grad=True))
        # recommender_ce_loss_div = torch.div(
        #     recommender_ce_loss,
        #     torch.tensor(float(self.train_batch_size), requires_grad=True))
        reweight_loss_theta_hat = \
            recommender_ce_loss / (recommender_ce_loss + self.reweight_loss_lambda * sup_con_curriculum_loss)
        self.reweight_loss_theta_after = self.reweight_loss_alpha * reweight_loss_theta_hat + \
                                         (1 - self.reweight_loss_alpha) * self.reweight_loss_theta_before
        self.reweight_loss_theta_before = self.reweight_loss_theta_after.data
        total_loss = recommender_ce_loss + self.reweight_loss_theta_after * sup_con_curriculum_loss

        # TODO 这里我没有使用flooding
        # 正则化flooding
        # total_loss_after_reg = (total_loss - self.total_loss_b).abs() + self.total_loss_b
        # return total_loss_after_reg
        return total_loss

    r"""这里是recommender推荐item的地方，这里用到了课程学习——从易到难，返回计算好的课程学习loss（batch mean）
    loss_list：重构loss
    drop_rate：课程学习的μ
    """

    def calculate_curriculum_loss(self, loss_list, drop_rate, seq_length):
        loss_list_sorted, loss_index_sorted = torch.sort(loss_list, descending=False, dim=-1)
        remind_rate = 1 - drop_rate
        remind_num = int(remind_rate * seq_length)
        recommender_ce_losses = loss_list_sorted[loss_list.shape[0] - seq_length: loss_list.shape[0] - seq_length + remind_num]
        recommender_ce_loss = recommender_ce_losses.mean()
        return recommender_ce_loss

    def predict(self, interaction):
        sub_model_seq_output, _, _, _, _ = self.forward(interaction, train_flag=False)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(sub_model_seq_output, test_item_emb).sum(dim=1)  # [B]

        return scores

    # 下面函数与recbole里面的代码很不一样
    def full_sort_predict(self, interaction):
        sub_model_seq_output, _, clean_seq_percent, user_interest_fusion_emb, soft_info_seq_emb = self.forward(interaction, train_flag=False)
        target_item = interaction[self.ITEM_ID]
        target_item_emb = self.item_embedding(target_item)

        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(sub_model_seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]

        denoised_user_emb_sim = torch.cosine_similarity(soft_info_seq_emb, target_item_emb, dim=-1).mean()
        nodenoised_user_emb_sim = torch.cosine_similarity(user_interest_fusion_emb, target_item_emb, dim=-1).mean()
        return scores, clean_seq_percent, denoised_user_emb_sim, nodenoised_user_emb_sim


class AttnReadout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            layer_norm=True,
            feat_drop=0.0,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_key_map = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_value_map = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_attention = nn.Linear(hidden_dim * 4,
                                      output_dim, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, item_seq_emb, user_interest_fusion_emb, mask):
        if self.layer_norm is not None:
            item_seq_emb = self.layer_norm(item_seq_emb)
        item_seq_emb = item_seq_emb * mask
        item_seq_emb = self.feat_drop(item_seq_emb)
        item_seq_emb = item_seq_emb * mask
        item_seq_emb_key = self.fc_key_map(item_seq_emb)
        item_seq_emb_key = item_seq_emb_key * mask
        user_interest_fusion_emb_expand = user_interest_fusion_emb.unsqueeze(1).expand_as(item_seq_emb)

        emb_subtract = item_seq_emb_key - user_interest_fusion_emb_expand
        emb_multiply = item_seq_emb_key * user_interest_fusion_emb_expand
        # emb_multiply = torch.matmul(
        #     item_seq_emb_key,
        #     user_interest_fusion_emb.unsqueeze(1).transpose(1, 2).to(torch.float))
        item_seq_two_dim_score = self.fc_attention(
            torch.cat((item_seq_emb_key, user_interest_fusion_emb_expand,
                       emb_subtract, emb_multiply), dim=-1)
        ) * mask

        score = self.softmax(item_seq_two_dim_score)

        denoised_items_emb = self.get_denoised_items_emb(
            item_seq_emb=item_seq_emb, score=score, mask=mask)

        return score, denoised_items_emb

    def get_denoised_items_emb(self, item_seq_emb, score, mask):
        item_seq_score = score[:, :, 0]
        item_seq_score = self.softmax(item_seq_score)
        item_seq_score = item_seq_score.unsqueeze(-1).expand_as(item_seq_emb)
        item_seq_emb_value = self.fc_value_map(item_seq_emb)
        item_seq_emb_value = item_seq_emb_value * mask
        denoised_items_emb = item_seq_score * item_seq_emb_value
        denoised_items_emb = denoised_items_emb * mask

        return denoised_items_emb

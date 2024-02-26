from recbole.trainer import Trainer
import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from transformers import WarmupLinearSchedule
from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage
from time import time
from utils.utils import indicators_20

class MsdcclTrainer(Trainer):

    def __init__(self, config, model):
        super(MsdcclTrainer, self).__init__(config, model)

        # 加入scheduler
        if self.config['scheduler']:
            warm_up_ratio = self.config['warm_up_ratio']
            total_train_examples = self.config['total_train_examples']
            train_batch_size = self.config['train_batch_size']
            total_steps = int(total_train_examples / train_batch_size)
            self.scheduler = WarmupLinearSchedule(
                self.optimizer,
                warmup_steps=warm_up_ratio * total_steps,
                t_total=total_steps)
        else:
            self.scheduler = False

        self.global_train_batches = 0
        self.gumbel_tau = self.config['gumbel_temperature']  # 初始化为0.5
        self.spu_cl_tau = self.config['supervised_contrastive_temperature']  # 初始化为0.2
        self.gumbel_tau_anneal = self.config['is_gumbel_tau_anneal']
        self.spu_cl_tau_anneal = self.config['is_spu_cl_tau_anneal']
        self.curriculum_learn_epoch = self.config['curriculum_learn_epoch']
        self.sigmoid_extent = self.config['sigmoid_extent']

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config['reg_weight'] and weight_decay and weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            if 'MSDCCL' in self.config['model']:
                optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
            else:
                optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    # 将下面的曲线改变了，不是原来的曲线了
    def drop_rate_schedule_curriculum(self, epoch_idx):
        # drop_rate = np.linspace(0.2 ** 1, 0, self.curriculum_learn_epoch)
        truncated_sigmoid = 1 / (1 + np.exp(-np.linspace(-self.sigmoid_extent, self.sigmoid_extent, self.curriculum_learn_epoch)))
        drop_rate = np.round(1 - truncated_sigmoid, 4) * 0.2

        if epoch_idx < self.curriculum_learn_epoch:
            return drop_rate[epoch_idx]
        else:
            return 0.0

    # 获得在退火时的温度
    def temperature_anneal(self, temperature, model):
        r = 1e-3
        if model == 'gumbel-softmax':
            temperature_after_anneal = max(1e-3, temperature * np.exp(- r * self.global_train_batches))
        elif model == 'supervised_contrastive_learning':
            temperature_after_anneal = max(0.1, temperature * np.exp(- r * self.global_train_batches))
        else:
            raise ValueError(f'Anneal model[{model}] is not supports.')

        return temperature_after_anneal

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()

        # 在这判断是否使用本模型的loss还是baseline模型的loss
        if 'MSDCCL' in str(self.model):
            loss_func = self.model.calculate_reweight_loss
        else:
            loss_func = self.model.calculate_loss

        total_loss = None
        r"""
        total_percent：一个epoch里面所有batch里面噪音item干净的程度（含有的噪音item越少，值越大）（单位batch）
        """
        total_percent = 0

        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=160,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            r"""
            下面的代码不同——原来没有这个变量global_train_batches
            这里记录global batches，为了对gumbel-softmax和supervised_contrastive_learning做退火
            这里是50个batch做一次退火
            """
            self.global_train_batches += 1
            if self.global_train_batches % 40 == 0:
                if self.gumbel_tau_anneal:
                    self.gumbel_tau = self.temperature_anneal(
                        temperature=self.gumbel_tau,
                        model='gumbel-softmax'
                    )
                if self.spu_cl_tau_anneal:
                    self.spu_cl_tau = self.temperature_anneal(
                        temperature=self.spu_cl_tau,
                        model='supervised_contrastive_learning'
                    )

            interaction = interaction.to(self.device)

            self.optimizer.zero_grad()

            r"""
            所以下面的语句就是为了获得loss
            clean_seq_percent：获得当前batch干净的程度
            drop_rate_for_curriculum：课程学习的μ
            """
            clean_seq_percent = 100
            if 'MSDCCL' in str(self.model):
                # with torch.autograd.set_detect_anomaly(True):
                #     losses, clean_seq_percent = loss_func(
                #         interaction,
                #         self.drop_rate_for_curriculum,
                #         self.gumbel_tau,
                #         self.spu_cl_tau)
                losses, clean_seq_percent = loss_func(
                    interaction,
                    self.drop_rate_for_curriculum,
                    self.gumbel_tau,
                    self.spu_cl_tau)
            else:
                losses = loss_func(interaction)
            total_percent += clean_seq_percent.__float__()

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            self._check_nan(loss)
            # with autograd.detect_anomaly():
            #     loss.backward()
            loss.backward()
            # for name, param in self.model.named_parameters():
            #     # 下面语句若有输出，则说明梯度里面有nan值
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print("nan gradient found")
            #         print("name:", name)
            #         print("param:", param.grad)
            #         raise SystemExit
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

            # self.tensorboard.add_scalar('tranin_total_loss', loss.item(), batch_idx)
            if self.gpu_available and show_progress:
                r"""
                total_loss：2个loss的总和
                clean_seq_percent：获得当前batch干净的程度
                gumbel_tau：gumbel-softmax的温度参数
                spu_cl_tau：supervised_contrastive_learning的温度参数
                drop_rate_for_curriculum：课程学习的drop_rate
                """
                iter_data.set_postfix_str(set_color('loss: ' + '%.2f' % loss.data.__float__(), 'yellow') + ', ' +
                                          set_color('clean_seq: ',
                                                    'blue') + '%.2f' % clean_seq_percent.__float__() + '%' + ', ' +
                                          set_color('gumbel_tau: ', 'blue') + '%.4f' % self.gumbel_tau + ', ' +
                                          set_color('spu_cl_tau: ', 'blue') + '%.4f' % self.spu_cl_tau + ', ' +
                                          set_color("curriculum's drop_rate: ",
                                                    'blue') + '%.3f' % self.drop_rate_for_curriculum
                                          )

        # 这里我没有使用scheduler模块
        if self.scheduler:
            # self.logger.info(set_color('Successfully utilize lr_scheduler strategy', 'pink'))
            self.scheduler.step()

        r"""
        下面与代码不同——原文只返回total_loss
        """
        return total_loss, total_percent / len(train_data)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            r"""
            下面语句更改了——多了一个输出参数
            clean_item_percent：整个epoch干净（无噪音item）的程度
            """
            self.drop_rate_for_curriculum = self.drop_rate_schedule_curriculum(epoch_idx)
            train_loss, clean_item_percent = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()

            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss / self.config['train_batch_size'],
                                                tag="Loss/Train-epoch")
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx},
                                         head='train')

            train_info = (set_color('total loss', 'blue') + ': %.4f' + '\n'
                          + set_color('clean sentence percentage', 'yellow') + ': %.2f' + '\n'
                          + set_color('curriculum learning drop-rate', 'yellow') + ': %.3f') % \
                         (train_loss, clean_item_percent, self.drop_rate_for_curriculum)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                r"""只是单纯做变量解释
                cur_step：the number of consecutive steps that did not exceed the best result
                stopping_step: threshold steps for stopping
                valid_metric_bigger (bool): whether the bigger the better
                """
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                r"""
                下面语句更改了——只是多了参数
                单纯的添加log信息
                train_loss: 一个epoch里面所有batch的loss（单位batch）
                clean_item_percent：整个epoch干净（无噪音item）的程度
                """
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f, " + set_color(
                            "cur_step", 'yellow') + ": %d]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score, self.cur_step)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(train_info + '\n' + valid_score_output + '\n' + valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)

        return self.best_valid_score, self.best_valid_result

    # positive_u: 应该是用户，positive_i: 应该是item
    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data

        # 记录句子的干净程度，clean_seq_percent
        clean_seq_percent = 100
        denoised_user_emb_sim = 1
        nodenoised_user_emb_sim = 1

        try:
            # Note: interaction without item ids
            if 'MSDCCL' in str(self.model):
                scores, clean_seq_percent, denoised_user_emb_sim, nodenoised_user_emb_sim = self.model.full_sort_predict(interaction.to(self.device))
            else:
                scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf

        return interaction, scores, positive_u, positive_i, clean_seq_percent, denoised_user_emb_sim, nodenoised_user_emb_sim

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        # 记录在测试集里面clean_seq_total和batch_counter的值
        clean_seq_total = 0
        denoised_user_emb_sim = 0
        nodenoised_user_emb_sim = 0
        batch_counter = 0

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        total_index = None
        total_positive = None
        total_seq_length = None

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i, clean_seq_percent,\
                denoised_user_emb_sim, nodenoised_user_emb_sim = eval_func(batched_data)

            clean_seq_total += clean_seq_percent
            denoised_user_emb_sim += denoised_user_emb_sim
            nodenoised_user_emb_sim += nodenoised_user_emb_sim
            batch_counter += 1
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)

            a, idx1 = torch.sort(scores, descending=True)#descending为alse，升序，为True，降序
            idx = idx1[:, :20]
            if total_index is None:
                total_index = idx.cpu().numpy()
                total_positive = positive_i.cpu().numpy()
                total_seq_length = batched_data[0]['item_length'].cpu().numpy()
            else:
                total_index = np.concatenate((total_index, idx.cpu().numpy()), axis=0)
                total_positive = np.concatenate((total_positive ,positive_i.cpu().numpy()), axis=0)
                total_seq_length = np.concatenate((total_seq_length ,batched_data[0]['item_length'].cpu().numpy()), axis=0)

        self.eval_collector.model_collect(self.model)

        # 下面是不同序列的hit，ndcg，mrr
        different_seq_res = indicators_20(total_index, total_positive, total_seq_length)

        # 下面是查看不同数据集长度分布
        seq_distribution = np.zeros(201)
        def fun(e):
            seq_distribution[e] += 1
            return 
        vfunc = np.vectorize(fun)
        result = vfunc(total_seq_length)

        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        denoised_num = (denoised_user_emb_sim / batch_counter)
        result['denoised'] = denoised_num.item() if torch.is_tensor(denoised_num) else denoised_num
        nodenoised_num = (nodenoised_user_emb_sim / batch_counter)
        result['nodenoised'] = nodenoised_num.item() if torch.is_tensor(nodenoised_num) else nodenoised_num

        self.wandblogger.log_eval_metrics(result, head='eval')

        return result

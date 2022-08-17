"""
@Time : 2022/8/17 16:59 
@Author : sunshb10145 
@File : sub_models.py 
@desc:
"""
from bert4keras.layers import *

from bert4keras.models import Transformer


class ReWeight(Layer):
    def __init__(self, init_reweight=0, trainable=True, **kwargs):
        super(ReWeight, self).__init__(**kwargs)
        self.init_reweight = init_reweight
        self.trainable = trainable

    def build(self, input_shape):
        initializer = keras.initializers.Constant(self.init_reweight)
        self.beta = self.add_weight(shape=(1,), initializer=initializer, name='reweight', trainable=self.trainable)

    def call(self, x):
        return x * self.beta


class SubTransformer(Transformer):
    def simplify(self, inputs):
        """过滤列表中的None"""
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            return inputs[0]
        return inputs

    def apply_attention(self, inputs, attention_name, arguments):
        # attention 层有很多变体，所以单独抽出来一个方法，来适应不同变体
        x = self.apply(inputs,
                       MultiHeadAttention,
                       name=attention_name,
                       head_nums=self.num_attention_heads,
                       head_size=self.attention_head_size,
                       arguments=arguments,
                       kernel_initializer=self.initializer,
                       with_residual_attention=self.residual_attention_scores)
        if self.residual_attention_scores:
            x, att_scores = x
            self.attention_scores = att_scores

        return x


class BERT(SubTransformer):
    """构建BERT模型
    """

    def __init__(
            self,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            hierarchical_position=None,  # 是否层次分解位置编码
            custom_position_ids=False,  # 是否自行传入位置id
            shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        self.shared_segment_embeddings = shared_segment_embeddings
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

    def get_inputs(self):
        """BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        inputs = [x_in]

        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Segment'
            )
            inputs.append(s_in)

        if self.custom_position_ids:
            p_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Position'
            )
            inputs.append(p_in)

        return inputs

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p = None
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.segment_vocab_size > 0:
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_bias': None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name='Pooler'
            )
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=pool_activation,
                kernel_initializer=self.initializer,
                name='Pooler-Dense'
            )
            if self.with_nsp:
                # Next Sentence Prediction部分
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=2,
                    activation='softmax',
                    kernel_initializer=self.initializer,
                    name='NSP-Proba'
                )
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense'
            )
            x = self.apply(
                inputs=self.simplify([x, z]),
                layer=LayerNormalization,
                conditional=(z is not None),
                hidden_units=self.layer_norm_conds[1],
                hidden_activation=self.layer_norm_conds[2],
                hidden_initializer=self.initializer,
                name='MLM-Norm'
            )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(
                inputs=x, layer=ScaleOffset, scale=False, name='MLM-Bias'
            )
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name='MLM-Activation'
            )
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(BERT, self).load_variable(checkpoint, name)
        if name in [
            'bert/embeddings/word_embeddings',
            'cls/predictions/output_bias',
        ]:
            return self.load_embeddings(variable)
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def create_variable(self, name, value, dtype=None):
        """在tensorflow中创建一个变量
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(BERT, self).create_variable(name, value, dtype)

    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


class ReZero(BERT):
    def __init__(self,
                 use_layernorm=None,  # None, pre, post, when None, then rezero
                 init_reweight=1.,  # init of reweight
                 reweight_trainable=True,
                 **kwargs,
                 ):
        super(ReZero, self).__init__(**kwargs)
        assert use_layernorm in [None, 'pre', 'post']
        self.use_layernorm = use_layernorm
        self.init_reweight = init_reweight
        self.reweight_trainable = reweight_trainable

    def apply_embeddings(self, inputs):
        """token_embedding + segment_embedding + position_embedding
        """
        x, s = inputs[:2]
        # condition layer norm
        z = self.layer_norm_conds[0]

        token_embedding = self.apply(inputs=x,
                                     layer=Embedding,
                                     name='Embedding-Token',
                                     input_dim=self.vocab_size,
                                     output_dim=self.embedding_size,
                                     embeddings_initializer=self.initializer,
                                     mask_zero=True
                                     )
        segment_embedding = self.apply(s,
                                       Embedding,
                                       name='Embedding-Segment',
                                       input_dim=self.segment_vocab_size,
                                       output_dim=self.embedding_size,
                                       embeddings_initializer=self.initializer,
                                       )
        token_with_seg = self.apply([token_embedding, segment_embedding], Add, name='Embedding-Token-Segment')
        x = self.apply(token_with_seg,
                       PositionEmbedding,
                       name='Embedding-Position',
                       input_dim=self.max_position,
                       output_dim=self.embedding_size,
                       embeddings_initializer=self.initializer,
                       merge_mode='add')

        # if pre layernorm, then delete layernorm in this block
        if self.use_layernorm != 'pre':
            x = self.apply(inputs=self.simplify([x, z]),
                           layer=LayerNormalization,
                           conditional=(z is not None),
                           condition_hidden_units=self.layer_norm_conds[1],
                           condition_hidden_activation=self.layer_norm_conds[2],
                           condition_hidden_initializer=self.initializer,
                           name='Embedding-Norm')
        x = self.apply(x,
                       Dropout,
                       name='Embedding-Dropout',
                       rate=self.dropout_rate)
        if self.hidden_size != self.embedding_size:
            x = self.apply(x,
                           Dense,
                           name='Embedding-Mapping',
                           units=self.hidden_size,
                           kernel_initializer=self.initializer)

        return x

    def apply_transformer_layers(self, inputs, idx):
        """
        post: Att --> Dropout --> Add --> LN --> FFN --> Dropout -->  Add --> LN
        pre: LN --> Att --> Dropout --> Add --> LN --> FFN --> Dropout --> Add
        rezero: Att --> ReWeight --> Dropout --> Add --> FFN -->ReWeight --> Dropout --> Add
        """
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % idx
        feed_forward_name = 'Transformer-%d-FeedForward' % idx
        attention_bias = self.compute_attention_bias(idx)

        x_pre, x = inputs, inputs
        z = self.layer_norm_conds[0]
        arguments = {'a_bias': None}
        if attention_bias is not None:
            arguments['a_bias'] = True
            x.append(attention_bias)

        if self.use_layernorm == 'pre':
            x = self.apply(inputs=self.simplify([x, z]),
                           layer=LayerNormalization,
                           conditional=(z is not None),
                           condition_hidden_units=self.layer_norm_conds[1],
                           condition_hidden_activation=self.layer_norm_conds[2],
                           condition_hidden_initializer=self.initializer,
                           name='%s-Norm' % attention_name,
                           )

        x = [x, x, x]
        # self-attention
        x = self.apply_attention(x, attention_name, arguments)

        # reweight residual-connection
        x = self.apply(x,
                       ReWeight,
                       name='%s-ReWeight' % attention_name,
                       init_reweight=self.init_reweight,
                       trainable=self.reweight_trainable
                       )

        x = self.apply(x,
                       Dropout,
                       name='%s-Dropout' % attention_name,
                       rate=self.dropout_rate)

        x = self.apply([x_pre, x],
                       Add,
                       name='%s-Add' % attention_name
                       )
        if self.use_layernorm == 'post':
            x = self.apply(inputs=self.simplify([x, z]),
                           layer=LayerNormalization,
                           conditional=(z is not None),
                           condition_hidden_units=self.layer_norm_conds[1],
                           condition_hidden_activation=self.layer_norm_conds[2],
                           condition_hidden_initializer=self.initializer,
                           name='%s-Norm' % attention_name,
                           )

        # feedforward
        x_pre = x
        if self.use_layernorm == 'pre':
            x = self.apply(inputs=self.simplify([x, z]),
                           layer=LayerNormalization,
                           conditional=(z is not None),
                           condition_hidden_units=self.layer_norm_conds[1],
                           condition_hidden_activation=self.layer_norm_conds[2],
                           condition_hidden_initializer=self.initializer,
                           name='%s-Norm' % feed_forward_name
                           )

        x = self.apply(x,
                       FeedForward,
                       name=feed_forward_name,
                       units=self.intermediate_size,
                       activation=self.hidden_act,
                       kernel_initializer=self.initializer
                       )

        # reweight residual-connection
        x = self.apply(x,
                       ReWeight,
                       name='%s-ReWeight' % feed_forward_name,
                       init_reweight=self.init_reweight,
                       trainable=self.reweight_trainable
                       )
        x = self.apply(x,
                       Dropout,
                       name='%s-Dropout' % feed_forward_name,
                       rate=self.dropout_rate)

        x = self.apply([x_pre, x],
                       Add,
                       name='%s-Add' % feed_forward_name)
        if self.use_layernorm == 'post':
            x = self.apply(inputs=self.simplify([x, z]),
                           layer=LayerNormalization,
                           conditional=(z is not None),
                           condition_hidden_units=self.layer_norm_conds[1],
                           condition_hidden_activation=self.layer_norm_conds[2],
                           condition_hidden_initializer=self.initializer,
                           name='%s-Norm' % feed_forward_name)

        return x

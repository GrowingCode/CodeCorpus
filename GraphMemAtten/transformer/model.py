import tensorflow as tf
from meta_info.hyper_parameter import d_head, n_head, n_layer,\
  d_embed, d_model, n_token, dropout, d_inner, dropatt, oracle_mem_len,\
  untie_r
from meta_info.non_hyper_constant import normal_initializer, top_ks, float_type,\
  int_type
from utils.initialize_util import random_normal_variable_initializer,\
  zero_variable_initializer


def positional_embedding(pos_seq, inv_freq, bsz=None):
  sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
  ''' ['tf.shape(sinusoid_inp):', [256 64]] '''
#   p_op = tf.print(['tf.shape(sinusoid_inp):', tf.shape(sinusoid_inp)])
#   with tf.control_dependencies([p_op]):
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  if bsz is not None:
    return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
  else:
    return pos_emb[:, None, :]


def rel_shift(x):
  x_size = tf.shape(input=x)
  x = tf.pad(tensor=x, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]])
  x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
  x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, x_size)
  return x


def _create_mask(qlen, mlen, same_length=False):
  attn_mask = tf.ones([qlen, qlen])
  mask_u = tf.linalg.band_part(attn_mask, 0, -1)
#   print(mask_u)
  mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
#   print(mask_dia)
  attn_mask_pad = tf.zeros([qlen, mlen])
  ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
  if same_length:
    mask_l = tf.linalg.band_part(attn_mask, -1, 0)
    ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
  return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
  if mem_len is None or prev_mem is None:
    new_mem = curr_out
  elif mem_len == 0:
    return prev_mem
  else:
    new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:]

  return tf.stop_gradient(new_mem)


class Transformer(tf.keras.Model):
  
  def __init__(self):
    super(Transformer, self).__init__()
    self.vocab_lookup_table = tf.Variable(random_normal_variable_initializer([n_token, d_embed]))
    self.proj_w = None
    if d_model != d_embed:
      self.proj_w = tf.Variable(random_normal_variable_initializer([d_embed, d_model]))
      
    self.token_output_w = tf.Variable(random_normal_variable_initializer([n_token, d_model]))
    self.token_output_softmax_b = tf.Variable(zero_variable_initializer([n_token]))
    
    self.transformer_dropout1 = tf.keras.layers.Dropout(dropout)
    self.transformer_dropout2 = tf.keras.layers.Dropout(dropout)
    self.transformer_dropout3 = tf.keras.layers.Dropout(dropout)
    
    if untie_r:
      self.r_w_bias = tf.Variable(random_normal_variable_initializer([n_layer, n_head, d_head]))
      self.r_r_bias = tf.Variable(random_normal_variable_initializer([n_layer, n_head, d_head]))
    else:
      self.r_w_bias = tf.Variable(random_normal_variable_initializer([n_head, d_head]))
      self.r_r_bias = tf.Variable(random_normal_variable_initializer([n_head, d_head]))
      
    self.t_layers = []
    for _ in range(n_layer):
      self.t_layers.append(TransformerLayer())
  
  def mask_adaptive_embedding_lookup(self, x):
    emb_scale = d_model ** 0.5
    proj_W = None
    y = tf.nn.embedding_lookup(self.vocab_lookup_table, x)
    if d_model != d_embed:
      proj_W = self.proj_w
      y = tf.einsum('ibe,ed->ibd', y, proj_W)
    
    y *= emb_scale
    return y
  
  def mask_adaptive_logsoftmax(self, hidden, target, valid_mask, compute_prediction,
                               return_mean=True):
    def _logit(x, W, b, proj):
      y = x
      if proj is not None:
        y = tf.einsum('ibd,ed->ibe', y, proj)
      return tf.einsum('ibd,nd->ibn', y, W) + b
  
    output = _logit(hidden, self.token_output_w, self.token_output_softmax_b, self.proj_w)
    
    nll = None
    if not compute_prediction:
      nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                           logits=output)
      print("n_token:" + str(n_token))
      target_max = tf.reduce_max(target)
      print("target_max:" + str(target_max))
      all_nll_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(nll), int_type))
      print("all_nll_nans:" + str(all_nll_nans))
    
#       print("nll:" + str(nll))
      ''' ['tf.shape(output):', [128 6 27]] '''
      ''' ['tf.shape(nll):', [128 6]] '''
      nll = nll * tf.cast(valid_mask, float_type)
      if return_mean:
        nll = tf.reduce_mean(input_tensor=nll)
      
    probs, predictions = None, None
    if compute_prediction:
      ''' ['tf.shape(predictions):', [128 6 top_ks[-1]]] '''
      t_probs = tf.math.log(tf.nn.softmax(output, axis=2))
      probs, predictions = tf.nn.top_k(t_probs, top_ks[-1])
      
    return probs, predictions, nll
  
  def transformer(self, dec_inp, target, mems, valid_mask, is_training, 
                  mem_len=oracle_mem_len, 
                  same_length=False, clamp_len=-1, 
                  untie_r=False):
    """
    cutoffs: a list of python int. Cutoffs for adaptive softmax.
    tie_projs: a list of python bools. Whether to tie the projections.
    use_tpu: if True, use one_hot in embedding lookup and bin-based implementation
          of adaptive softmax.
    perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
          Only used in the adaptive setting.
    """
    
    '''' one example: ['dec_inp[:,0]:', [6 10 13 ... 6 21 1], 'target[:,0]:', [10 13 2 ... 21 1 7]] '''
    ''' ['same_length:', False] '''
    
    new_mems = []
    
    ''' ['tf.shape(r_r_bias):', [2 16]] '''
    ''' ['tf.shape(r_w_bias):', [2 16]] '''
    
    ''' ['tf.shape(dec_inp):', [128 6]] '''
    ''' ['tf.shape(target):', [128 6]] '''
    
    qlen = tf.shape(input=dec_inp)[0]
    mlen = tf.shape(input=mems[0])[0] if mems is not None else 0
    
    ''' ['qlen:', 128, 'mlen:', 128] '''
    
#     p_op_dec_inp = tf.print(['tf.shape(dec_inp):', tf.shape(dec_inp), 'tf.shape(target):', tf.shape(target)])
#     p_op_target = tf.print(['tf.shape(r_r_bias):', tf.shape(r_r_bias)])
#     with tf.control_dependencies([p_op_dec_inp]):
    klen = mlen + qlen
    
    ''' ['klen:', 256] '''
    
    embeddings = self.mask_adaptive_embedding_lookup(x=dec_inp)
    
    ''' shared_params[0] is vocab_table '''
    ''' ['tf.shape(embeddings):', [128 6 128], 'tf.shape(shared_params[0]):', [27 128]] '''
    
    attn_mask = _create_mask(qlen, mlen, same_length)
    
    ''' ['tf.shape(attn_mask):', [128 256]] '''
    
    pos_seq = tf.range(klen - 1, -1, -1.0)
    if clamp_len > 0:
      pos_seq = tf.minimum(pos_seq, clamp_len)
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    ''' ['clamp_len:', -1, 'tf.shape(pos_seq):', [256], 'tf.shape(inv_freq):', [64]] '''
    ''' ['pos_seq:', [255 254 253 252 251 250 249 248 247 246 245 244 243 242 241 240 239 238 237 236 235 234 233 232 231 230 229 228 227 226 225 224 223 222 221 220 219 218 217 216 215 214 213 212 211 210 209 208 207 206 205 204 203 202 201 200 199 198 197 196 195 194 193 192 191 190 189 188 187 186 185 184 183 182 181 180 179 178 177 176 175 174 173 172 171 170 169 168 167 166 165 164 163 162 161 160 159 158 157 156 155 154 153 152 151 150 149 148 147 146 145 144 143 142 141 140 139 138 137 136 135 134 133 132 131 130 129 128 127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112 111 110 109 108 107 106 105 104 103 102 101 100 99 98 97 96 95 94 93 92 91 90 89 88 87 86 85 84 83 82 81 80 79 78 77 76 75 74 73 72 71 70 69 68 67 66 65 64 63 62 61 60 59 58 57 56 55 54 53 52 51 50 49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]] '''
    ''' ['inv_freq:', [1 0.865964353 0.749894202 0.649381638 0.562341332 0.486967534 0.421696514 0.365174145 0.316227764 0.273841977 0.237137362 0.2053525 0.177827939 0.153992653 0.133352146 0.115478203 0.1 0.0865964293 0.0749894157 0.0649381652 0.0562341288 0.0486967526 0.0421696492 0.0365174115 0.0316227786 0.0273841955 0.0237137359 0.0205352511 0.0177827943 0.0153992651 0.0133352149 0.0115478197 0.01 0.00865964312 0.00749894232 0.00649381662 0.00562341325 0.00486967526 0.00421696482 0.00365174143 0.00316227786 0.00273841969 0.00237137382 0.00205352507 0.00177827943 0.00153992651 0.00133352145 0.00115478202 0.001 0.000865964335 0.000749894185 0.000649381662 0.000562341302 0.000486967503 0.000421696488 0.000365174114 0.000316227786 0.000273841957 0.000237137385 0.000205352495 0.00017782794 0.000153992645 0.00013335215 0.000115478193]] '''
#     p_op = tf.print(['pos_seq:', pos_seq], summarize = 300)
#     with tf.control_dependencies([p_op]):
    pos_emb = positional_embedding(pos_seq, inv_freq)
    
    ''' ['tf.shape(pos_emb):', [256 1 128]] '''
    
#     p_op = tf.print(['tf.shape(mems):', tf.shape(mems)])
#     with tf.control_dependencies([p_op]):
    output = self.transformer_dropout1(embeddings, training=is_training)
    ''' ['tf.shape(output):', [128 6 128]] '''
    pos_emb = self.transformer_dropout2(pos_emb, training=is_training)
    ''' ['tf.shape(pos_emb):', [256 1 128]] '''

    if mems is None:
      mems = [None] * n_layer
    ''' ['tf.shape(mems):', [3 128 6 128]] '''
    
    for i in range(n_layer):
      # cache new mems
      new_mems.append(_cache_mem(output, mems[i], mem_len))
      
      t_layer = self.t_layers[i]
      output = t_layer.rel_multihead_attn(
          w=output,
          r=pos_emb,
          r_w_bias=self.r_w_bias if not untie_r else self.r_w_bias[i],
          r_r_bias=self.r_r_bias if not untie_r else self.r_r_bias[i],
          attn_mask=attn_mask,
          mems=mems[i],
          is_training=is_training)
      output = t_layer.positionwise_FF(
          inp=output,
          is_training=is_training)

    output = self.transformer_dropout3(output, training=is_training)
    
    ''' ['tf.shape(output):', [128 6 128], 'tf.shape(target):', [128 6]] '''
    probs, predictions, loss = self.mask_adaptive_logsoftmax(
        hidden=output,
        target=target,
        valid_mask=valid_mask,
        compute_prediction=(is_training <= 0))
    
    return output, probs, predictions, loss, new_mems
  
  def get_token_output_parameters(self):
    return self.token_output_w
  

class TransformerLayer():
  
  def __init__(self):
    self.pos_wise_ff_dense1 = tf.keras.layers.Dense(d_inner, activation=tf.nn.relu, kernel_initializer=normal_initializer)
    self.pos_wise_ff_dropout1 = tf.keras.layers.Dropout(dropout)
    self.pos_wise_ff_dense2 = tf.keras.layers.Dense(d_model, activation=tf.nn.relu, kernel_initializer=normal_initializer)
    self.pos_wise_ff_dropout2 = tf.keras.layers.Dropout(dropout)
    self.pos_wise_ff_ln = tf.keras.layers.LayerNormalization()
    
    self.multi_atten_dense1 = tf.keras.layers.Dense(3 * n_head * d_head, use_bias=False, kernel_initializer=normal_initializer)
    self.multi_atten_dense2 = tf.keras.layers.Dense(n_head * d_head, use_bias=False, kernel_initializer=normal_initializer)
    self.multi_atten_dropout1 = tf.keras.layers.Dropout(dropatt)
    self.multi_atten_dense3 = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=normal_initializer)
    self.multi_atten_dropout2 = tf.keras.layers.Dropout(dropout)
    self.multi_atten_ln = tf.keras.layers.LayerNormalization()
    
  def positionwise_FF(self, inp, is_training=True):
    output = self.pos_wise_ff_dense1(inp)
    output = self.pos_wise_ff_dropout1(output, training=is_training)
    output = self.pos_wise_ff_dense2(output)
    output = self.pos_wise_ff_dropout2(output, training=is_training)
    ''' output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1) '''
    output = self.pos_wise_ff_ln(output + inp)
    return output
  
  def rel_multihead_attn(self, w, r, r_w_bias, r_r_bias, attn_mask, mems, is_training):
    ''' w is the embedding of input x, r is the position embedding '''
    ''' ['tf.shape(w):', [128 6 128], 'tf.shape(r):', [256 1 128]] '''
    ''' ['tf.shape(r_w_bias):', [2 16], 'tf.shape(r_r_bias):', [2 16]] '''
    ''' ['tf.shape(attn_mask):', [128 256], 'tf.shape(mems):', [128 6 128]] '''
    ''' ['d_model:', 128, 'n_head:', 2, 'd_head:', 16] '''
    
    scale = 1 / (d_head ** 0.5)
    qlen = tf.shape(input=w)[0]
    rlen = tf.shape(input=r)[0]
    bsz = tf.shape(input=w)[1]

    cat = tf.concat([mems, w],
                    0) if mems is not None and mems.shape.ndims > 1 else w
    
    ''' ['tf.shape(cat):', [256 6 128]] '''
    
    w_heads = self.multi_atten_dense1(cat)
    
    ''' ['tf.shape(w_heads):', [256 6 96]] '''
    
    r_head_k = self.multi_atten_dense2(r)
    
    ''' ['tf.shape(r_head_k):', [256 1 32]] '''
    
    w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
    
    ''' ['tf.shape(w_head_q):', [256 6 32], 'tf.shape(w_head_k):',[256 6 32],'tf.shape(w_head_v):',[256 6 32]] '''
    
    w_head_q = w_head_q[-qlen:]
    
    ''' ['tf.shape(w_head_q):',[128 6 32]] '''
    
    klen = tf.shape(input=w_head_k)[0]

    w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
    w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
    w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

    ''' ['tf.shape(w_head_q):',[128 6 2 16],'tf.shape(w_head_k):',[256 6 2 16],'tf.shape(w_head_v):',[256 6 2 16]] '''

    r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])
    
    ''' ['tf.shape(r_head_k):',[256 2 16]] '''
    
    rw_head_q = w_head_q + r_w_bias
    rr_head_q = w_head_q + r_r_bias

    AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
    BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
    BD = rel_shift(BD)
    
    ''' ['tf.shape(AC):', [128 256 6 2], 'tf.shape(BD):', [128 256 6 2]] '''
    
    attn_score = (AC + BD) * scale
    attn_mask_t = attn_mask[:, :, None, None]
    
    ''' ['tf.shape(attn_score):', [128 256 6 2], 'tf.shape(attn_mask_t):', [128 256 1 1]] '''
    
    attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = self.multi_atten_dropout1(attn_prob, training=is_training)

    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
    
    ''' ['tf.shape(attn_score):',[128 256 6 2],'tf.shape(attn_prob):',[128 256 6 2],'tf.shape(attn_vec):',[128 6 2 16]] '''
    
    size_t = tf.shape(input=attn_vec)
    attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])
    
    ''' ['tf.shape(attn_vec):', [128 6 32]] '''
    
    attn_out = self.multi_atten_dense3(attn_vec)
    attn_out = self.multi_atten_dropout2(attn_out, training=is_training)

    ''' ['tf.shape(attn_out):', [128 6 128]] '''
#     output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
#     p_op = tf.print(['tf.shape(attn_out):', tf.shape(attn_out)])
#     with tf.control_dependencies([p_op]):
    output = self.multi_atten_ln(attn_out + w)
    return output
  
  





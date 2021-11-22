# UtilsTools

### The often used basic functions for CNNs and other modules.

1. #### This is the comparison of the listed 6 activation functions.

   Include sigmoid, h-sigmoid, ReLU, swish, h-swish, LeakyReLU.
   
    | Activation function | Mathematics Expression |
    | ------------------- | ---------------------- |
    | Sigmoid             | 1/(1+exp(-x))          |
    | H-sigmoid           | ReLU6(x+3)/6           |
    | ReLU                | Max(x, 0)              |
    | Swish               | x*1/(1+exp(-x))        |
    | H-Swish             | x*ReLU6(x+3)/6         |
    |LeakyReLU            | Max(0, x)+slope*Min(0, x)|            |


### 

![comparison](comparison.jpg)


cls_token: torch.Size([1, 1, 768])
embedding.weight: torch.Size([768, 3, 16, 16])
embedding.bias: torch.Size([768])
transformer.pos_embedding.pos_embedding: torch.Size([1, 257, 768])
transformer.encoder_layers.0.layer_norm1.weight: torch.Size([768])
transformer.encoder_layers.0.layer_norm1.bias: torch.Size([768])
transformer.encoder_layers.0.multi_head_att.query.weight: torch.Size([768, 12, 64])
transformer.encoder_layers.0.multi_head_att.query.bias: torch.Size([12, 64])
transformer.encoder_layers.0.multi_head_att.key.weight: torch.Size([768, 12, 64])
transformer.encoder_layers.0.multi_head_att.key.bias: torch.Size([12, 64])
transformer.encoder_layers.0.multi_head_att.value.weight: torch.Size([768, 12, 64])
transformer.encoder_layers.0.multi_head_att.value.bias: torch.Size([12, 64])
transformer.encoder_layers.0.multi_head_att.out.weight: torch.Size([12, 64, 768])
transformer.encoder_layers.0.multi_head_att.out.bias: torch.Size([768])
transformer.encoder_layers.0.layer_norm2.weight: torch.Size([768])
transformer.encoder_layers.0.layer_norm2.bias: torch.Size([768])
transformer.encoder_layers.0.mlp.fc1.weight: torch.Size([3072, 768])
transformer.encoder_layers.0.mlp.fc1.bias: torch.Size([3072])
transformer.encoder_layers.0.mlp.fc2.weight: torch.Size([768, 3072])
transformer.encoder_layers.0.mlp.fc2.bias: torch.Size([768])
transformer.encoder_layers.1.layer_norm1.weight: torch.Size([768])
transformer.encoder_layers.1.layer_norm1.bias: torch.Size([768])
transformer.encoder_layers.1.multi_head_att.query.weight: torch.Size([768, 12, 64])
transformer.encoder_layers.1.multi_head_att.query.bias: torch.Size([12, 64])
transformer.encoder_layers.1.multi_head_att.key.weight: torch.Size([768, 12, 64])
transformer.encoder_layers.1.multi_head_att.key.bias: torch.Size([12, 64])
transformer.encoder_layers.1.multi_head_att.value.weight: torch.Size([768, 12, 64])
transformer.encoder_layers.1.multi_head_att.value.bias: torch.Size([12, 64])
transformer.encoder_layers.1.multi_head_att.out.weight: torch.Size([12, 64, 768])
transformer.encoder_layers.1.multi_head_att.out.bias: torch.Size([768])
transformer.encoder_layers.1.layer_norm2.weight: torch.Size([768])
transformer.encoder_layers.1.layer_norm2.bias: torch.Size([768])
transformer.encoder_layers.1.mlp.fc1.weight: torch.Size([3072, 768])
transformer.encoder_layers.1.mlp.fc1.bias: torch.Size([3072])
transformer.encoder_layers.1.mlp.fc2.weight: torch.Size([768, 3072])
transformer.encoder_layers.1.mlp.fc2.bias: torch.Size([768])
transformer.norm.weight: torch.Size([768])
transformer.norm.bias: torch.Size([768])
classifier.weight: torch.Size([1000, 768])
classifier.bias: torch.Size([1000])

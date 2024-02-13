主要训练文件为train.py和train_improve.py, 测试文件为test_model.ipynb

其中train.py 为原论文提供的训练代码，运行方式以及参数如下

python train.py --data-train dataset/NYC/NYC_train.csv --data-val dataset/NYC/NYC_val.csv  --time-units 48 --time-feature norm_in_day_time --poi-embed-dim 128 --user-embed-dim 128 --time-embed-dim 32 --cat-embed-dim 32 --node-attn-nhid 128 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2 --batch 16 --epochs 200 --name exp1

train_improve.py 为改进后的训练代码，运行方式以及参数如下

python train_improve.py --data-train dataset/NYC/NYC_train.csv --data-val dataset/NYC/NYC_val.csv  --time-units 48 --time-feature norm_in_day_time --poi-embed-dim 128 --user-embed-dim 128 --time-embed-dim 32 --cat-embed-dim 32 --node-attn-nhid 128 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2 --batch 16 --epochs 200 --name exp1

test_model.ipynb 为模型测试代码，包含了用户相似度挖掘，请顺序运行代码块。

除此之外，test_worldcloud.py 为相似用户词云图展示的代码，similar_users_2.csv为相似用户的文件

NYC_train.csv等数据集在dataset/NYC目录下
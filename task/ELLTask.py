from model import BertClassifier
from utils import logger_init
from data import Dataset
from data.ELLDataset import TestDataset
from transformers import BertConfig
from torch.optim import lr_scheduler
import logging
import torch
import os
import pandas as pd
import time
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'train.csv')
        #self.val_file_path = os.path.join(self.dataset_dir, 'val.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'test.csv')
        self.sub_file_path = os.path.join(self.dataset_dir,"sample_submission.csv")
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.batch_size = 20
        self.learning_rate = 3.5e-5
        self.epochs = 3
        self.nums_labels = 6
        self.labels = ['cohesion',
          'syntax',
          'vocabulary',
          'phraseology',
          'grammar' ,
          'conventions'
          ]


        logger_init(log_file_name='ELL', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")

def datadeal(config):
    df = pd.read_csv(config.train_file_path)
    logging.info(df['full_text'])

def train(model,config):
    df = pd.read_csv(config.train_file_path)
    logging.info(df.head())
    train_data, val_data, test_data = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])
    print(len(train_data),len(val_data),len(test_data))

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, config.batch_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500,eta_min=1e-6)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # 开始进入训练循环
    for epoch_num in range(config.epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(config.device).float()
            mask = train_input['attention_mask'].to(config.device)
            input_id = train_input['input_ids'].squeeze(1).to(config.device)
            print(input_id.shape)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失

        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(config.device)
                mask = val_input['attention_mask'].to(config.device)
                input_id = val_input['input_ids'].squeeze(1).to(config.device)
                output = model(input_id, mask)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

    logging.info(f'''Epochs: {epoch_num + 1} 
                  | Train Loss: {total_loss_train / len(train_data): .3f} 
                  | Val Loss: {total_loss_val / len(val_data): .3f} ''')
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    torch.save(model.state_dict(), model_save_path)

def predict(model,config):
    test_data = pd.read_csv(config.test_file_path)
    test = TestDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    prediction = []
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        model.eval()
    if use_cuda:
        model = model.cuda()
    with torch.no_grad():
        for test_input in test_dataloader:

            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            logging.info(output)
            for pred in output.cpu():
                prediction.append(np.array([min(max(1.0, i), 5.0) for i in np.array(pred)]))
    return np.array(prediction)

def submission(prediction,config):
    submission = pd.read_csv(config.sub_file_path)
    submission.cohesion = prediction[:, 0]
    submission.syntax = prediction[:, 1]
    submission.vocabulary = prediction[:, 2]
    submission.phraseology = prediction[:, 3]
    submission.grammar = prediction[:, 4]
    submission.conventions = prediction[:, 5]
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    model_config = ModelConfig()
    # 定义bert分类器模型
    #model = BertClassifier(model_config)
    #train(model,model_config)
    #predictions=predict(model,model_config)
    #submission(predictions,model_config)
    datadeal(model_config)
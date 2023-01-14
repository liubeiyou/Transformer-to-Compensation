import torch
import xlwt, xlrd
from xlutils.copy import copy
from my_dataset import *
from model.vit_model import Transformer
from torch.utils.data import DataLoader
from multi_train_utils.train_eval_utils import *
import openpyxl as op


def op_toExcel(data, fileName):  # openpyxl library for storing data to excel
    wb = op.Workbook()  # Create the workbook object
    ws = wb['Sheet']  # Creating child tables
    ws.append(['Temperature', 'Humidity', 'Concentration','Value'])  # Adding Table headers
    for i in range(data.size()[0]):
        d = data[i][0].item(), data[i][1].item(), data[i][2].item(),data[i][3].item()
        ws.append(d)   # Write one line at a time
    wb.save(fileName)

def up(data):
    up_data = nn.Upsample(scale_factor=(1,100.4083),mode='bilinear')
    return up_data(data)

def test(filename):
    ture_data = Excel_dataset_test(filename)
    ture_data_loader = torch.utils.data.DataLoader(ture_data,
                                                    batch_size=240,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=0)
    for step , batch in enumerate(ture_data_loader):
        data1 = batch.unsqueeze(0).unsqueeze(0)
        data1 = data1.permute(0,1,3,2)
        data1 = up(data1).squeeze(0).squeeze(0).permute(1,0)
        fileName = ''
        op_toExcel(data1, fileName)





###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    filename = ""
    test(filename)  # recall test function

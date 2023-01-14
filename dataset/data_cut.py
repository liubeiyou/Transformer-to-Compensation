'''
Instructions: The purpose of this script is to randomly split the data into training and test sets,
              with the proportion adjusted according to the size of the data
'''
from my_dataset import *
from torch.utils.data import DataLoader
from multi_train_utils.train_eval_utils import *
import openpyxl as op
###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################


def op_toExcel(data, fileName):  # openpyxl library for storing data to excel
    wb = op.Workbook()  # Create the workbook object
    ws = wb['Sheet']  # Creating child tables
    ws.append(['Temperature', 'Humidity', 'Concentration','Value'])  # Adding Table headers
    for i in range(data.size()[0]):
        d = data[i][0].item(),data[i][1].item(),data[i][2].item(),data[i][3].item()
        ws.append(d)  # Write one line at a time
    wb.save(fileName)

def test(file_path):
    test_data = Excel_dataset_test(file_path)
    train_data_set,valid_data_set=data_split(test_data,0.8)
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=19211,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(valid_data_set,
                                             batch_size=4811,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=0)

    for step, batch in enumerate(train_loader):  # 100  3

        fileName = 'D:\practice\python\Trans\data/train1.xlsx'
        op_toExcel(batch, fileName)

    for step, data1 in enumerate(val_loader):
        fileName = 'D:\practice\python\Trans\data/val1.xlsx'
        op_toExcel(data1, fileName)



###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "D:\practice\python\Trans\data\data_pred.xlsx"
    test(file_path)  # recall test function

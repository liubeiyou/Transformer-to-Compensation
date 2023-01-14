from dataset.my_dataset import *
from torch.utils.data import DataLoader
from multi_train_utils.train_eval_utils import *
import openpyxl as op

###################################################################
# ------------------- Main pred (Run second) -------------------
###################################################################

ckpt = ""  # chose model
device = torch.device('cuda')
model = Transformer().to(device)
weights = torch.load(ckpt, map_location=device)  # Loading the model on multiple Gpus
weights_dict = {}  # Models saved by multiple Gpus have the module prefix
for k, v in weights.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v
model.load_state_dict(weights_dict)


def op_toExcel(data, fileName):  # openpyxl library for storing data to excel
    wb = op.Workbook()  # Create the workbook object
    ws = wb['Sheet']  # Creating child tables
    ws.append(['Temperature', 'Humidity', 'Concentration','Prediction'])  # Adding Table headers
    for i in range(data.size()[0]):
        d = data[i][0].item(), data[i][1].item(), data[i][2].item(), data[i][3].item()
        ws.append(d)  # Write one line at a time
    wb.save(fileName)


def test(file_path):
    pred_data = Excel_dataset_test(file_path)
    pred_data_loader = torch.utils.data.DataLoader(pred_data,
                                                   batch_size=256,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=0)
    T = torch.zeros([1, 1])
    for step, batch in enumerate(pred_data_loader):  # 100  3
        with torch.no_grad():
            data= batch
            data = data.to(device).unsqueeze(1)
            pred = model(data)
            pred = pred.unsqueeze(1)
            data = data.squeeze(1)
            if step == 0:
                data = torch.cat([data,pred], dim=1)
                T = data
            else:
                data = torch.cat([data,pred], dim=1)
                T = torch.cat([T, data], dim=0)
    T = T.squeeze(1)
    fileName = ''
    op_toExcel(T, fileName)


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = ""  # data need to  be predicted
    test(file_path)  # recall test function

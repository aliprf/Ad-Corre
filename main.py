from test_model import TestModels
from train import TrainModel
from config import DatasetName, DatasetType
if __name__ == "__main__":

    '''testing the pre-trained models'''
    tester = TestModels(h5_address='./trained_models/AffectNet_6336.h5')
    tester.recognize_fer(img_path='./img.jpg')

    '''training part'''
    trainer = TrainModel(dataset_name=DatasetName.affectnet, ds_type=DatasetType.train_7)
    trainer.train(arch="xcp", weight_path="./")
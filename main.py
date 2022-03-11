from test_model import TestModels

if __name__ == "__main__":


    tester = TestModels(h5_address='./trained_models/AffectNet_6336.h5')

    tester.recognize_fer(img_path='./img.jpg')
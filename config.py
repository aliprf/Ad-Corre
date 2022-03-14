class DatasetName:
    affectnet = 'affectnet'
    rafdb = 'rafdb'
    fer2013 = 'fer2013'


class ExpressionCodesRafdb:
    Surprise = 1
    Fear = 2
    Disgust = 3
    Happiness = 4
    Sadness = 5
    Anger = 6
    Neutral = 7

class ExpressionCodesAffectnet:
    neutral = 0
    happy = 1
    sad = 2
    surprise = 3
    fear = 4
    disgust = 5
    anger = 6
    contempt = 7
    none = 8
    uncertain = 9
    noface = 10

class DatasetType:
    train = 0
    train_7 = 1
    eval = 2
    eval_7 = 3
    test = 4


class LearningConfig:
    # batch_size = 100
    batch_size = 50
    # batch_size = 1
    # batch_size = 5
    virtual_batch_size = 5* batch_size
    epochs = 300
    embedding_size = 256 # we have 360 filters at the end
    # embedding_size = 128 # we have 360 filters at the end
    labels_history_frame = 1000
    num_classes = 7
    # num_embeddings = 7
    num_embeddings = 10


class InputDataSize:
    image_input_size = 224


class FerPlusConf:
    _prefix_path = './FER_plus'  # --> zeue

    orig_image_path_train = _prefix_path + '/orig/train/'
    orig_image_path_test = _prefix_path + '/orig/test/'

    '''only 7 labels'''
    no_aug_train_img_path = _prefix_path + '/train_set/images/'
    no_aug_train_annotation_path = _prefix_path + '/train_set/annotations/'

    aug_train_img_path = _prefix_path + '/train_set_aug/images/'
    aug_train_annotation_path = _prefix_path + '/train_set_aug/annotations/'
    aug_train_masked_img_path = _prefix_path + '/train_set_aug/masked_images/'

    weight_save_path = _prefix_path + '/weight_saving_path/'

    '''both public&private test:'''
    test_img_path = _prefix_path + '/test_set/images/'
    test_annotation_path = _prefix_path + '/test_set/annotations/'
    test_masked_img_path = _prefix_path + '/test_set/masked_images/'

    '''private test:'''
    private_test_img_path = _prefix_path + '/private_test_set/images/'
    private_test_annotation_path = _prefix_path + '/private_test_set/annotations/'
    '''public test-> Eval'''
    public_test_img_path = _prefix_path + '/public_test_set/images/'
    public_test_annotation_path = _prefix_path + '/public_test_set/annotations/'


class RafDBConf:
    _prefix_path = './RAF-DB'  #

    orig_annotation_txt_path = _prefix_path + '/list_patition_label.txt'
    orig_image_path = _prefix_path + '/original/'
    orig_bounding_box = _prefix_path + '/boundingbox/'

    '''only 7 labels'''
    no_aug_train_img_path = _prefix_path + '/train_set/images/'
    no_aug_train_annotation_path = _prefix_path + '/train_set/annotations/'

    aug_train_img_path = _prefix_path + '/train_set_aug/images/'
    aug_train_annotation_path = _prefix_path + '/train_set_aug/annotations/'
    aug_train_masked_img_path = _prefix_path + '/train_set_aug/masked_images/'

    test_img_path = _prefix_path + '/test_set/images/'
    test_annotation_path = _prefix_path + '/test_set/annotations/'
    test_masked_img_path = _prefix_path + '/test_set/masked_images/'

    augmentation_factor = 5

    weight_save_path = _prefix_path + '/weight_saving_path/'


class AffectnetConf:
    """"""
    '''atlas'''
    _prefix_path = './affectnet'  # --> Aq

    orig_csv_train_path = _prefix_path + '/orig/training.csv'
    orig_csv_evaluate_path = _prefix_path + '/orig/validation.csv'

    '''8 labels'''
    no_aug_train_img_path = _prefix_path + '/train_set/images/'
    no_aug_train_annotation_path = _prefix_path + '/train_set/annotations/'

    aug_train_img_path = _prefix_path + '/train_set_aug/images/'
    aug_train_annotation_path = _prefix_path + '/train_set_aug/annotations/'
    aug_train_masked_img_path = _prefix_path + '/train_set_aug/masked_images/'

    eval_img_path = _prefix_path + '/eval_set/images/'
    eval_annotation_path = _prefix_path + '/eval_set/annotations/'
    eval_masked_img_path = _prefix_path + '/eval_set/masked_images/'

    '''7 labels'''
    no_aug_train_img_path_7 = _prefix_path + '/train_set_7/images/'
    no_aug_train_annotation_path_7 = _prefix_path + '/train_set_7/annotations/'

    aug_train_img_path_7 = _prefix_path + '/train_set_7_aug/images/'
    aug_train_annotation_path_7 = _prefix_path + '/train_set_7_aug/annotations/'
    aug_train_masked_img_path_7 = _prefix_path + '/train_set_7_aug/masked_images/'

    eval_img_path_7 = _prefix_path + '/eval_set_7/images/'
    eval_annotation_path_7 = _prefix_path + '/eval_set_7/annotations/'
    eval_masked_img_path_7 = _prefix_path + '/eval_set_7/masked_images/'

    weight_save_path = _prefix_path + '/weight_saving_path/'

    num_of_samples_train = 2420940
    num_of_samples_train_7 = 0
    num_of_samples_eval = 3999
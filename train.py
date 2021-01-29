import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from tqdm import tqdm

import config as c
from localization import export_gradient_maps
from model import DifferNet, save_model, save_weights, save_parameters, save_roc_plot, MaskDifferNet
from utils import *
from operator import itemgetter
import cv2

class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))


def train(train_loader, validate_loader):
    model = DifferNet()
    optimizer = torch.optim.Adam([{'params': model.nf.parameters()}


        ], lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)

    save_name_pre = '{}_{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}'.format(c.modelname, c.rotation_degree,
                                               c.crop_top, c.crop_left, c.crop_bottom, c.crop_right)

    score_obs = Score_Observer('AUROC')

    for epoch in range(c.meta_epochs):

        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()
                inputs, labels = preprocess_batch(data)  # move to device and reshape
                # TODO inspect
                # inputs += torch.randn(*inputs.shape).cuda() * c.add_img_noise

                z = model(inputs)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                train_loss.append(t2np(loss))
                loss.backward()
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if c.verbose:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        if not (validate_loader is None):
            # evaluate
            model.eval()
            if c.verbose:
                print('\nCompute loss and scores on validate set:')
            test_loss = list()
            test_z = list()
            test_labels = list()
            with torch.no_grad():
                for i, data in enumerate(tqdm(validate_loader, disable=c.hide_tqdm_bar)):
                    inputs, labels = preprocess_batch(data)
                    z = model(inputs)
                    loss = get_loss(z, model.nf.jacobian(run_forward=False))
                    test_z.append(z)
                    test_loss.append(t2np(loss))
                    test_labels.append(t2np(labels))

            test_loss = np.mean(np.array(test_loss))

            test_labels = np.concatenate(test_labels)
            is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

            z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
            anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
            AUROC = roc_auc_score(is_anomaly, anomaly_score)
            score_obs.update(AUROC, epoch,
                            print_score=c.verbose or epoch == c.meta_epochs - 1)

            fpr, tpr, thresholds = roc_curve(is_anomaly, anomaly_score)
            model_parameters = {}
            model_parameters['fpr'] = fpr.tolist()
            model_parameters['tpr'] = tpr.tolist()
            model_parameters['thresholds'] = thresholds.tolist()
            model_parameters['AUROC'] = AUROC

            save_parameters(model_parameters, save_name_pre + "_{:.4f}".format(AUROC))
            save_roc_plot(fpr, tpr, save_name_pre + "_{:.4f}".format(AUROC))

            if c.verbose:
                print('Epoch: {:d} \t validate_loss: {:.4f}'.format(epoch, test_loss))

                # compare is_anomaly and anomaly_score
                np.set_printoptions(precision=2, suppress=True)
                print('is_anomaly:    ', is_anomaly)
                print('anomaly_score: ', anomaly_score)
                print('fpr:           ', fpr)
                print('tpr:           ', tpr)
                print('thresholds:    ', thresholds)

    if c.grad_map_viz and not (validate_loader is None):
        export_gradient_maps(model, validate_loader, optimizer, 1)

    if c.save_model:
        model.to('cpu')
        save_model(model, save_name_pre + '.pth')
        save_weights(model, save_name_pre + '.weights.pth')

    return model, model_parameters

def test(model, model_parameters, test_loader):
    print("Running test")
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04,
                                 weight_decay=1e-5)
    # score_obs = Score_Observer('AUROC')
    # evaluate
    model.to(c.device)
    model.eval()
    # print(f"model={model}")
    epoch = 0
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_loss = list()
    test_z = list()
    test_labels = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            # inputs = Variable(inputs, requires_grad=True)
            print(f"i={i}: labels={labels}, size of inputs={inputs.size()}")
            # print(f"inputs={inputs}")
            z = model(inputs)
            # print(f"z={z}")
            loss = get_loss(z, model.nf.jacobian(run_forward=False))
            test_z.append(z)
            test_loss.append(t2np(loss))
            test_labels.append(t2np(labels))

        test_loss = np.mean(np.array(test_loss))
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
    # score_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
    #                 print_score=c.verbose or epoch == c.meta_epochs - 1)

    # get the threshold for target true positive rate
    for i in range(len(model_parameters['tpr'])):
        if model_parameters['tpr'][i] > c.target_tpr:
            target_threshold = model_parameters['thresholds'][i]
            break

    is_anomaly_detected = np.array([0 if l < target_threshold else 1 for l in anomaly_score])

    # calculate test accuracy
    error_count = 0
    for i in range(len(is_anomaly)):
        if is_anomaly[i] != is_anomaly_detected[i]:
            error_count += 1

    test_accuracy = 1 - float(error_count) / len(is_anomaly)

    print(f"test_labels={test_labels}, is_anomaly={is_anomaly},anomaly_score={anomaly_score},is_anomaly_detected={is_anomaly_detected}")
    print(f"target_tpr={c.target_tpr}, target_threshold={target_threshold}, test_accuracy={test_accuracy}")

def predict(model, model_parameters, predict_loader):
    print("Predicting")
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    model.eval()
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_z = list()
    test_labels = list()
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(predict_loader):
            inputs, labels = preprocess_batch(data)
            if c.frame_name_is_given:
                frame = int(predict_loader.dataset.imgs[i][0].split('frame',1)[1].split('-')[0])
            frame = i
            #print(f"i={i}: frame#={frame}, labels={labels.cpu().numpy()[0]}, size of inputs={inputs.size()}")
            predictions.append([frame, predict_loader.dataset.imgs[i][0], labels.cpu().numpy()[0], 0, 0])
            z = model(inputs)
            test_z.append(z)
            test_labels.append(t2np(labels))

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))

    for i in range(len(model_parameters['tpr'])):
        if model_parameters['tpr'][i] > c.target_tpr:
            target_threshold = model_parameters['thresholds'][i]
            break

    is_anomaly_detected = []
    i = 0
    for l in anomaly_score:
        predictions[i][4] = l
        if l < target_threshold:
            is_anomaly_detected.append(0)
            predictions[i][3] = 0
        else:
            is_anomaly_detected.append(1)
            predictions[i][3] = 1
        i += 1
    predictions = sorted(predictions, key=itemgetter(0))

    # calculate test accuracy
    error_count = 0
    for i in range(len(is_anomaly)):
        if is_anomaly[i] != is_anomaly_detected[i]:
            error_count += 1

    test_accuracy = 1 - float(error_count) / len(is_anomaly)

    for i in range(len(predictions)):
        msg = 'frame: ' + str(i) + '. '
        if (predictions[i][3] == 1):
            msg += 'prediction: defective. '
        else:
            msg += 'prediction: good. '

        if (predictions[i][2] == 1):
            msg += 'ground truth: defective. '
        else:
            msg += 'ground truth: good. '

        msg += 'anomaly score: ' + str(round(predictions[i][4], 4)) + '. '
        msg += 'threshold: ' + str(round(target_threshold, 4)) + '. '
        msg += 'accuracy: ' + str(round(test_accuracy * 100, 2)) + '%'

        print(msg)

    # print(f"test_labels={test_labels}, is_anomaly={is_anomaly},anomaly_score={anomaly_score},is_anomaly_detected={is_anomaly_detected}")
    print(f"target_tpr={c.target_tpr}, target_threshold={target_threshold}, test_accuracy={test_accuracy}")
    if c.grad_map_viz:
        print("saving gradient maps...")
        export_gradient_maps(model, predict_loader, optimizer, -1)



    # visualize the prediction result
    if c.visualization:
        for i in range(len(predictions)):
            # load file path
            file_path = predictions[i][1]
            idx = file_path.index('video')
            file_path = file_path[:idx] + 'original-' + file_path[idx:]
            file_path = file_path.replace("predict\\test", "zerobox-2010-1-original")

            # rotate and resize image
            img = cv2.imread(file_path)
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.resize(img, (600, 900))

            # display prediction on each frame
            font = cv2.FONT_HERSHEY_DUPLEX
            font_size = 0.65
            pos_x = 330
            if (predictions[i][3] == 1):
                img = cv2.putText(img, 'prediction: defective', (pos_x, 810), font,
                                  font_size, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                img = cv2.putText(img, 'prediction: good', (pos_x, 810), font,
                                  font_size, (0, 255, 0), 1, cv2.LINE_AA)

            if (predictions[i][2] == 1):
                img = cv2.putText(img, 'ground truth: defective', (pos_x, 830), font,
                                  font_size, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                img = cv2.putText(img, 'ground truth: good', (pos_x, 830), font,
                                  font_size, (0, 255, 0), 1, cv2.LINE_AA)

            img = cv2.putText(img, 'anomaly score: ' + str(round(predictions[i][4], 4)), (pos_x, 850), font,
                              font_size, (0, 255, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, 'threshold: ' + str(round(target_threshold, 4)), (pos_x, 870), font,
                              font_size, (0, 255, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, 'accuracy: ' + str(round(test_accuracy * 100, 2)) + '%', (pos_x, 890), font,
                              font_size, (0, 255, 0), 1, cv2.LINE_AA)
            # show results
            cv2.imshow('window', img)
            cv2.waitKey(220)

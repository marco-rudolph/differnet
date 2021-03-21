from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from localization import export_gradient_maps
from model import save_roc_plot
from utils import *
from operator import itemgetter
import cv2

def test(model, model_parameters, test_loader):
    print("Running test")
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    model.eval()
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_z = list()
    test_labels = list()
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = preprocess_batch(data)
            if c.frame_name_is_given:
                frame = int(test_loader.dataset.imgs[i][0].split('frame',1)[1].split('-')[0])
            frame = i
            #print(f"i={i}: frame#={frame}, labels={labels.cpu().numpy()[0]}, size of inputs={inputs.size()}")
            predictions.append([frame, test_loader.dataset.imgs[i][0], labels.cpu().numpy()[0], 0, 0])
            z = model(inputs)
            test_z.append(z)
            test_labels.append(t2np(labels))

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
    AUROC = roc_auc_score(is_anomaly, anomaly_score)
    fpr, tpr, thresholds = roc_curve(is_anomaly, anomaly_score)
    save_roc_plot(fpr, tpr, c.modelname + "_{:.4f}_test".format(AUROC))

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
    #todo: tpr/fpr display.

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
        export_gradient_maps(model, test_loader, optimizer, -1)

    # visualize the prediction result
    if c.visualization:
        for i in range(len(predictions)):
            # load file path
            file_path = predictions[i][1]
            idx = file_path.index('video')
            file_path = file_path[:idx] + 'original-' + file_path[idx:]
            file_path = file_path.replace("test\\test", "zerobox-2010-1-original")

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
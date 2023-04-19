# Python imports

# Third-party imports


# Package imports




def evalute_calibrated_model(calibrated_model, base_model, x_train, y_train,
                             x_test, y_test, ae_names, output_dir):

    logging.info(f'Predicting calibrated and uncalibrated for train and test')
    calibrated_train_predictions = calibrated_model.predict(x_train)
    calibrated_test_predictions = calibrated_model.predict(x_test)
    uncalibrated_train_predictions = base_model.predict(x_train)
    uncalibrated_test_predictions = base_model.predict(x_test)

    def evaluate_submodel(model_idx):

        # Evaluate model

        ae_name = ae_names[model_idx]
        ae_eval_path = Path(output_dir) / ae_name
        ae_eval_path.mkdir(exist_ok=True)

        logging.info(f'Evaluating {ae_name} model and generating figures')
        submodel_calibrated_train_predictions = calibrated_train_predictions[:,
                                                                             model_idx]
        submodel_calibrated_test_predictions = calibrated_test_predictions[:,
                                                                           model_idx]

        submodel_uncalibrated_train_predictions = uncalibrated_train_predictions[:,
                                                                                 model_idx]
        submodel_uncalibrated_test_predictions = uncalibrated_test_predictions[:,
                                                                               model_idx]

        # Plot roc curves
        plot_roc("Train",
                 y_train[:, model_idx],
                 submodel_calibrated_train_predictions,
                 color=COLORS[0])
        plot_roc("Test",
                 y_test[:, model_idx],
                 submodel_calibrated_test_predictions,
                 color=COLORS[1],
                 linestyle='--')
        plt.title(f'ROC Curve ({ae_name})')
        plt.legend(loc='lower right')
        plt.savefig(ae_eval_path / 'roc_curve.png')

        plt.clf()

        # Calibration Curve

        n_bins = 10
        x_cal_train, y_cal_train = calibration_curve(
            y_train[:, model_idx],
            submodel_calibrated_train_predictions,
            n_bins=n_bins,
            strategy='quantile')

        x_cal_test, y_cal_test = calibration_curve(
            y_test[:, model_idx],
            submodel_calibrated_test_predictions,
            n_bins=n_bins,
            strategy='quantile')

        x_uncal_train, y_uncal_train = calibration_curve(
            y_train[:, model_idx],
            submodel_uncalibrated_train_predictions,
            n_bins=n_bins,
            strategy='quantile')

        x_uncal_test, y_uncal_test = calibration_curve(
            y_test[:, model_idx],
            submodel_uncalibrated_test_predictions,
            n_bins=n_bins,
            strategy='quantile')

        plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally Calibrated')

        plot_calibration_curve("Calibrated Train",
                               x_cal_train,
                               y_cal_train,
                               fmt='r-')
        plot_calibration_curve("Calibrated Test",
                               x_cal_test,
                               y_cal_test,
                               fmt='b-')
        plot_calibration_curve("Uncalibrated Train",
                               x_uncal_train,
                               y_uncal_train,
                               fmt='r--')
        plot_calibration_curve("Uncalibrated Test",
                               x_uncal_test,
                               y_uncal_test,
                               fmt='b--')

        leg = plt.legend(loc='upper left')
        plt.xlabel('Average Predicted Probability in each bin')
        plt.ylabel('Ratio of positives')
        plt.title(f'Calibration Curve ({ae_name})')
        plt.savefig(ae_eval_path / 'calibration_curve.png')

        plt.clf()

        plot_prc("Train",
                 y_train[:, model_idx],
                 submodel_calibrated_train_predictions,
                 color=COLORS[0])
        plot_prc("Test",
                 y_test[:, model_idx],
                 submodel_calibrated_test_predictions,
                 color=COLORS[1],
                 linestyle='--')
        plt.title(f'PRC Curve ({ae_name})')
        plt.legend(loc='lower right')
        plt.savefig(ae_eval_path / 'prc_curve.png')
        plt.clf()

        logging.info('Done.')

    # Loop through each output
    for i in range(y_train.shape[1]):
        evaluate_submodel(i)


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    #   plt.xlim([-0.5,20])
    #   plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()


#   ax.set_aspect('equal')


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(
        labels, predictions)

    plt.plot(recall, precision, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_calibration_curve(name, x_cal, y_cal, fmt=None):
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally Calibrated')

    plt.plot(y_cal, x_cal, fmt, marker='.', label=f'LSTM Classifier ({name})')

    # leg = plt.legend(loc='upper left')
    # plt.xlabel('Average Predicted Probability in each bin')
    # plt.ylabel('Ratio of positives')
    #     plt.xlim(0, 0.2)
    #     plt.ylim(0, 0.2)

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


class LossReporter:
    def __init__(self, nb_epochs, report_validation=True):
        self._nb_epochs = nb_epochs
        self._epochs = []
        self._train_losses = []
        self._val_losses = []
        self._current_lines = None
        self._report_validation = report_validation

        # plt.ion()
        plt.clf()
        plt.axis([0, self._nb_epochs, 0, 2.5])
        plt.grid()

    def __call__(self, epoch, train_loss, val_loss):
        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        if self._report_validation:
            self._val_losses.append(val_loss)

        if self._current_lines:
            for l in self._current_lines:
                l.remove()

        line1 = plt.plot(self._epochs, self._train_losses, 'r-', label='Train loss')
        if self._report_validation:
            line2 = plt.plot(self._epochs, self._val_losses, 'b-', label='Validation loss')
            self._current_lines = [line1[0], line2[0]]
        else:
            self._current_lines = [line1[0]]
        plt.legend(loc='upper right')

        plt.pause(0.0001)
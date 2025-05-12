from pathlib import Path

class Logger:
    """ Extracts and/or persists tracker information. """

    def __init__(self, path: str = None):
        """
        Parameters
        ----------
        path : str or Path, optional
            Path to where data will be stored.
        """
        path = Path("runs") if path is None else Path(path)
        self.path = path.expanduser().resolve()

    def on_epoch_start(self, epoch: int, **kwargs):
        """Actions to take on the start of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        """Actions to take on the end of an epoch."""
        pass

    def on_iter_start(self, epoch: int, update: int, tag: str, **kwargs):
        """Actions to take on the start of an iteration."""
        pass

    def on_iter_update(self, epoch: int, update: int, tag: str, **kwargs):
        """Actions to take when an update has occurred."""
        pass

    def on_iter_end(self, epoch: int, update: int, tag: str, **kwargs):
        """Actions to take on the end of an iteration."""
        pass

    def on_update(self, epoch: int, update: int):
        """Actions to take when the model is updated."""
        pass

class Tracker:
    """ Tracks useful information as learning progresses. """

    def __init__(self, *loggers: "Logger"):
        """
        Parameters
        ----------
        logger0, logger1, ... loggerN : Logger
            One or more loggers for logging training information.
        """
        self.epoch = 0
        self.update = 0
        self._tag = None
        self._losses = []
        self._summary = {}

        self.loggers = list(loggers)

    def start_epoch(self, count: bool = True):
        """ Start one iteration of updates over the training data. """
        if count:
            self.epoch += 1

        self._summary.clear()
        for logger in self.loggers:
            logger.on_epoch_start(self.epoch)

    def end_epoch(self):
        """ Wrap up one iteration of updates over the training data. """
        for logger in self.loggers:
            logger.on_epoch_end(self.epoch, **self._summary)

        return dict(self._summary)

    def start(self, tag: str, num_batches: int = None):
        """ Start a loop over mini-batches. """
        self._tag = tag
        self._losses.clear()
        for logger in self.loggers:
            logger.on_iter_start(self.epoch, self.update, self._tag, num_steps_expected=num_batches)

    def step(self, loss: float):
        """ Register the loss of a single mini-batch. """
        self._losses.append(loss)
        for logger in self.loggers:
            logger.on_iter_update(self.epoch, self.update, self._tag, loss=loss)

    def summary(self):
        """ Wrap up and summarise a loop over mini-batches. """
        losses = self._losses
        avg_loss = float("nan") if len(losses) == 0 else sum(losses) / len(losses)
        self._summary[self._tag] = avg_loss
        for logger in self.loggers:
            logger.on_iter_end(self.epoch, self.update, self._tag, avg_loss=avg_loss)

        return avg_loss

    def count_update(self):
        """ Increase the update counter. """
        self.update += 1
        for logger in self.loggers:
            logger.on_update(self.epoch, self.update)


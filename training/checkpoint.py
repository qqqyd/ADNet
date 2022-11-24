from concern.config import Configurable, State
import os
import torch


class Checkpoint(Configurable):
    start_epoch = State(default=0)
    start_iter = State(default=0)
    resume = State()
    only_load_backbone = State(default=False)
    load_test = State(default=False)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']
        if 'only_load_backbone' in cmd:
            self.only_load_backbone = cmd['only_load_backbone']
        if 'load_test' in cmd:
            self.load_test = cmd['load_test']

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            self.logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        logger.info("Resuming from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)
        if self.only_load_backbone:
            logger.info("Only load backbone")
            backbone_dict = {k: v for k, v in state_dict.items() if 'backbone' in k}
            model.load_state_dict(backbone_dict, strict=False)
        elif self.load_test:
            logger.info("Load test")
            state_dict = {k: v for k, v in state_dict.items() if k != 'model.module.decoder.dilate_pred.0.weight'}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter

from ultralytics import YOLO #model
from ultralytics import yolo #librairy

from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, TQDM_BAR_FORMAT, callbacks,
                                    is_git_dir, yaml_load, colorstr)
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, attempt_load_one_weight)


import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist

class DetectionTrainerCustom(yolo.v8.detect.DetectionTrainer):
    def do_pass():
        pass

    #could overide if needed : l199 yolo/engine/trainer
    #def _setup_train(self, world_size): 

    #surement probleme de files avec __init__

    #THIS override
    def _do_train(self, world_size=1):
        """
        overide training step to custom train
        """
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        #before epochs
        #print("rank", RANK) #-1
        BACKBONE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        HEAD_FIRST = [12, 15]
        HEAD_SECOND = [16, 18, 19, 21]
        #freeze backbone and first part of head
        for name, param in self.model.named_parameters():
            layer_num = int(name.split(".")[1])
            if layer_num in BACKBONE or layer_num in HEAD_FIRST:
                param.requires_grad = False
                print(f"param {name} has requ_grad : {param.requires_grad}")

        #FIX LR (2times smaller than final lr) :  + dont call lr_sceduler
        for g in self.optimizer.param_groups:
            g['lr'] = 5e-5
        
        #training loop
        for epoch in range(0, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()

            #reduce lr
            if epoch == self.epochs//3*2:
                print("reduce lr to 1e-5")
                for g in self.optimizer.param_groups:
                    g['lr'] = 1e-5

            #unfreeze some layers
            if epoch == self.epochs//3:
                print("unfreeze head1 at ", epoch)
                for name, param in self.model.named_parameters():
                    layer_num = int(name.split(".")[1])
                    if layer_num in HEAD_FIRST:
                        param.requires_grad = True
                        print(f"param {name} has requ_grad : {param.requires_grad}")
            # dont touh backbone : gives worse results
            # if epoch == self.epochs//3*2:
            #     print("unfreeze beackbone at ", epoch)
            #     for name, param in self.model.named_parameters():
            #         layer_num = int(name.split(".")[1])
            #         if layer_num in BACKBONE:
            #             param.requires_grad = True
            #             print(f"param {name} has requ_grad : {param.requires_grad}")
                


            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch['img'])
                    self.loss, self.loss_items = self.criterion(preds, batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            #self.scheduler.step() #don't change lr
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')


#mapping for YOLOCustom : class not used
TASK_MAP = {
    'classify': [
        ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
        yolo.v8.classify.ClassificationPredictor],
    'detect': [
        DetectionModel, DetectionTrainerCustom, yolo.v8.detect.DetectionValidator, #custom trainer
        yolo.v8.detect.DetectionPredictor],
}

class YOLOCustom(YOLO):
    """
    override of some fuctions to perform our custom training
    """
    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
            these are : 
            cfg 
            data : 'coco.yaml'
            'imgsz': 640
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
            self.session.check_disk_space()
        check_pip_update_available()
        overrides = self.overrides.copy()
        #breakpoint()
        overrides.update(kwargs)
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
    
        self.task = overrides.get('task') or self.task
        self.trainer = TASK_MAP[self.task][1](overrides=overrides) #NEED HERE TO REDIFINE THE TRAINER
        if not overrides.get('resume'):  # manually set model only if not resuming
            #load the pretrained weights and model with requ_grad = True
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP





if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt',
                        choices=('yolov8n.pt', 'yolov8m.pt', 'yolov8x.pt'),
                            help='size of model to use')
    parser.add_argument('--epochs', default=3, type=int,
                                 help='number of epochs')
    args = parser.parse_args()


    model = args.model
    print("loading yolo, model: ", model)

    model = YOLOCustom('trained_models/' + model) #pretrained

    

    # data_folder = "/work/nmuenger_trinca/annotations/" #real
    #data_folder = "/work/vita/nmuenger_trinca/annotations_reduced/" #for tests
    #in yolo data dataloader stream loader l.180 : added this dir (hardcoded) :(

    #NEED to test with data side to model (also slooooow ?)

    #---PREDICTIONS---
    #pred = model(data_folder + "train.txt") #file that say where the images are
    #print(pred)
    #print(pred[0].boxes.cls)
    #print(pred[0].boxes.xywhn)


    #check if everythink is on gpu : YES yolo/engine/trainer.py l171

    model.train(data = "tsr_dataset.yaml", epochs=args.epochs)